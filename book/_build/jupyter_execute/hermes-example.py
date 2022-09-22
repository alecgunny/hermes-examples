#!/usr/bin/env python
# coding: utf-8

# # Inference-as-a-Service with `hermes`
# This is intended to serve as a brief overview as to how the `hermes` libraries might be used to accelerate an inference deployment as-a-service using NVIDIA's [Triton inference server](https://developer.nvidia.com/nvidia-triton-inference-server). We'll start by showing what a vanilla, suboptimal deployment might look like to introduce all the relevant concepts, then make things slightly more complex to show how to analyze and identify the bottlenecks in a deployment and remove them.
# 
# As you can see from the `pyproject.toml`, all of the relevant `hermes` libraries are currently installed in this environment. In a production setting, you might consider breaking these up to keep environments more lightweight. For example, `hermes.quiver` might be installed in your training environment to export at train time, or might be installed in a dedicated export deployment if it involves more complex dependencies like TensorRT. Meanwhile, your inference environment might have `hermes.aeriel` and `hermes.stillwater` installed for deploying and monitoring an inference service. This is not critical to the discussion here, but worth bringing up to point out that the `hermes` libraries are not a monolith and are intended to be lightweight and composable.
# 
# ## Overview
# In this example, we'll begin by building a neural network, then exporting it from memory to disk in a format that Triton can use for serving it for inference. Obviously, in practice we'd like to train this network on some data between these steps, but since we're more interested here in the inference side, we'll pretend this training has been done elsewhere. For the sake of simplicity, we'll build a 1D convolutional network with a single output (which might be used for e.g. for binary classification).
# 
# Once our model has been properly exported, we'll spin up a Triton inference service which will load the model and expose it for inference via gRPC requests. We'll then build some dummy inference data and iterate through it to build requests to send to our inference service, aggregating its responses into a timeseries of network outputs.
# 
# We'll start with our imports. From `hermes`, we'll be using
# - `hermes.quiver` to handle exporting our model to a format usable by Triton
# - `hermes.aeriel.serve` to spin up an inference service locally using Python APIs
# - `hermes.aeriel.client` to make requests to that inference service
# - `hermes.stillwater.ServerMonitor` to keep track of server-side inference metrics

# In[1]:


import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ratelimiter import RateLimiter

# our hermes imports
from hermes import quiver as qv
from hermes.aeriel.client import InferenceClient
from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor

# couple cheap local utilities
from src import utils
from src import plotting

logger = utils.get_logger()


# Now let's establish some parameters for both the model we'd like to build as well as for our inference time deployment.

# In[2]:


# model parameters
NUM_IFOS = 2  # number of interferometers analyzed by our model
SAMPLE_RATE = 2048  # rate at which input data to the model is sampled
KERNEL_LENGTH = 4  # length of the input to the model in seconds

# inference parameters
INFERENCE_DATA_LENGTH = 2048  # amount of data to analyze at inference time
INFERENCE_SAMPLING_RATE = 0.25  # rate at which we'll sample input windows from the inference data
INFERENCE_RATE = 250  # seconds of data we'll try to analyze per second

# convert some of these into more useful units for slicing purposes
kernel_size = int(SAMPLE_RATE * KERNEL_LENGTH)
inference_stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
inference_data_size = int(SAMPLE_RATE * INFERENCE_DATA_LENGTH)
num_inferences = (inference_data_size - kernel_size) // inference_stride + 1

# limit the number of requests we make per second
# so that we don't overload the network or server
kernels_per_second = int(INFERENCE_RATE * INFERENCE_SAMPLING_RATE)
rate_limiter = RateLimiter(max_calls=kernels_per_second, period=1)


# Now let's build our extremely simple network

# In[3]:


class GlobalAvgPool(torch.nn.Module):
    def forward(self, x):
        return x.mean(axis=-1)


nn = torch.nn.Sequential(
    torch.nn.Conv1d(NUM_IFOS, 8, kernel_size=7, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(8, 32, kernel_size=7, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(32, 64, kernel_size=7, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(64, 128, kernel_size=7, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(128, 256, kernel_size=7, stride=2),
    torch.nn.ReLU(),
    GlobalAvgPool(),
    torch.nn.Linear(256, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 1)
)

# INSERT TRAINING CODE HERE


# Ok, now the set-up work is done. We have a "trained" neural network, and we're ready to export it for inference. Enter `hermes`. One of the key concepts in as-a-service inference is the idea of a **model repository**, a local (or cloud-based) directory with a prescribed structure that hosts all the versions of all the models to be served for inference.
# 
# Maintaining this prescribed structure, as well as the configs that Triton needs to be able to map to named inputs and outputs, can be onerous and non-trivial, so `hermes.quiver` was built to take the headache out of building and maintaining these repositories. In this next step, we'll build a model repository (clearing it beforehand in case you run this notebook multiple times), add a new entry to it for the network that we just build, then export the current version of this network to that repository as an [ONNX](https://onnx.ai/) binary which Triton can load and serve for inference.

# In[4]:


# let's make sure we're starting with a fresh repo
repo_path = "model-repo"
utils.clear_repo(repo_path)

# initialize a blank model repository
repo = qv.ModelRepository(repo_path)
assert len(repo.models) == 0  # this attribute will get updated as we add models

# create a new entry in the repo for our model
model = repo.add("my-classifier", platform=qv.Platform.ONNX)
assert len(repo.models) == 1
assert model == repo.models["my-classifier"]

# now export our current version of the network to this entry.
# Since we haven't exported any versions of this model yet,
# Triton needs to know what names to give the inputs and
# outputs and what shapes to expect, so we have to specify
# them explicitly this first time.
# Note that -1 indicates variable length batch dimension.
model.export_version(
    nn,
    input_shapes={"hoft": (-1, NUM_IFOS, kernel_size)},
    output_names=["prob"]
)


# Note that our model is associated with a `Config` object that describes the metadata Triton requires

# In[5]:


model.config


# and that each model can be associated with multiple different versions corresponding to different weight values, or even wholesale different architectures. The only thing that matters is that the network represents the same input-to-output mapping:

# In[6]:


model.versions


# And with that, our model is ready to be served for inference! In the code below, we'll use `hermes.aeriel.serve` to spin up a [Singularity container](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) on a single GPU (index 0) which will run a Triton inference service in the background. Once the `serve` context exits, the service and the container running it will both be spun-down.
# 
# Once the server is up and running, we'll use `hermes.aeriel.client` to establish a client connection to it, then iterate through our dummy data and make requests using this connection. The requests are made asynchronously and the responses parsed in a callback thread. The parsed responses are made available in the main thread through a queue which can be succinctly accessed by `InferenceClient.get()`, which will return `None` if there are no responses to be returned.
# 
# Note that below, we'll do our inference in batch sizes of 1. In principle, we could get better throughput by increasing that batch size (at the cost of some latency), but as we'll see momentarily there are more pressing issues we need to address first.

# In[7]:


# Start by spinning up an inference service.
# Note that this will print the singularity command
# used to start the service. This is unfortunately an
# unavoidable bug in singularity right now (if you don't
# want to receive an unnecessary warning instead), but it's
# probably good to note what is really happening under the
# hood anyway.
# The `instance` returned by the context is an object
# representing the running singularity container instance.
with serve("model-repo", gpus=[0]) as instance:
    # Do our data generation in parallel while the server
    # spins up. This will obviously be pretty fast, but
    # for more complicated data generation steps this can
    # be a good way to parallelize your efforts.
    hoft = np.random.randn(NUM_IFOS, inference_data_size).astype("float32")

    # now wait until the inference service is online and
    # ready to receive requests before we attempt to connect
    logger.info("Waiting for inference service to come online...")
    instance.wait()
    logger.info("Service ready!")

    # establish a client connection to the inference service
    # and infer the names and shapes of the inputs it expects
    client = InferenceClient(
        "localhost:8001",
        model_name="my-classifier",
        model_version=1  # can use -1 to imply latest version
    )

    # client context establishes a streaming connection to
    # the inference service. Since we're not yet streaming
    # updates but making requests individually, we don't
    # technically need this, but it's a good habit to get into.
    with client:
        # now iterate through our inference timeseries at the
        # prescribed stride and send these inputs to the server
        # for inference.
        for i in range(int(num_inferences)):
            start = i * inference_stride
            stop = start + kernel_size

            # add a dummy dimension for the batch
            kernel = hoft[:, start: stop][None]
            with rate_limiter:
                client.infer(kernel)

        # now that we've submitted all our inference requests,
        # start pulling them from the output queue as they
        # become available.
        results = []
        while len(results) < num_inferences:
            response = client.get()
            if response is not None:
                y, request_id, sequence_id = response
                results.append(y[:, 0])
        logger.info("Inference complete!")

# concatenate all the responses into a single timeseries
results = np.concatenate(results)


# In principle, that does it: we served a model, ran data through it, and from looking at the timestamps on the logs it looks like we roughly hit our inference rate target. Younow have everything you need to run inference-as-a-service with `hermes`. But there are some considerations you might think about that could help improve both network performance and throughput. For some inspiration, let's take a look at what our timeseries of network responses looks like:

# In[8]:


p = plotting.plot_timeseries(results, INFERENCE_SAMPLING_RATE, KERNEL_LENGTH)
plotting.show(p)


# So as we should have expected, it's just a timeseries of more or less random data. What is worth noting about this timeseries, however, is the rate at which it is sampled: 0.25 Hz means that for shorter-duration events like binary blackhole mergers, we only get to make one prediction on each event. Surely there might be some benefit to taking predictions from multiple overlapping windows containing the same event and aggregating them somewhow. Let's increase our inference sampling rate to, say, 4 Hz and see how this impacts our throughput.
# 
# For this next round of inference, I'm going to add a little more complication up front at the expense of slightly more elegance at inference time. Rather than iterating through responses after all our inference requests have been submitted, I'll set up a callback up front that aggregates our responses into an array in real-time in the callback thread, then returns the array once completed.

# In[9]:


class Callback:
    def __init__(self, num_inferences):
        self.y = np.zeros((num_inferences,))

    def __call__(self, response, request_id, sequence_id):
        self.y[request_id] = response[0, 0]
        if (request_id + 1) == len(self.y):
            return self.y


# reset some of our parameters with a new inference sampling rate
INFERENCE_SAMPLING_RATE = 4
inference_stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
num_inferences = (inference_data_size - kernel_size) // inference_stride + 1
kernels_per_second = int(INFERENCE_RATE * INFERENCE_SAMPLING_RATE)

rate_limiter = RateLimiter(max_calls=kernels_per_second / 20, period=0.05)
callback = Callback(num_inferences)

# from here things will look more or less the same
with serve("model-repo", gpus=[0]) as instance:
    logger.info("Waiting for inference service to come online...")
    instance.wait()
    logger.info("Service ready!")

    # instantiate client with custom callback
    client = InferenceClient(
        "localhost:8001",
        model_name="my-classifier",
        model_version=1,
        callback=callback
    )

    with client:
        for i in range(int(num_inferences)):
            start = i * inference_stride
            stop = start + kernel_size
            kernel = hoft[:, start: stop][None]

            # pass explicit request ids this time
            # for the callback to use
            with rate_limiter:
                client.infer(kernel, request_id=i)

        # now wait until the callback returns its
        # filled out array to the client's queue
        while True:
            results = client.get()
            if results is not None:
                logger.info("Inference complete!")
                break


# And now let's take a look at how this timeseries looks

# In[10]:


p = plotting.plot_timeseries(results, INFERENCE_SAMPLING_RATE, KERNEL_LENGTH)
plotting.show(p)


# So things work with higher frequency inference, but a quick eyeball of our logs indicate that we're now falling well short of our intended inference rate target (at time of writing, I'm seeing a time delta of ~43s, which translates to an inference rate of ~48 seconds of data / s). Why is that? Is our network capable of handling the desired number of requests per second, or is there something else going on?
# 
# For diagnosing these sorts of questions, Triton includes a secondary service which makes some inference metrics available (served at port 8002 by default). Clients can query this service for per-model information like cumulative time spent queueing and executing requests, as well as cumulative inference counts. The `hermes.stillwater` library has a `ServerMonitor` class which, in a separate process, queries this service and organizes the returned metrics from potentially many servers and writes them to a local log file.
# 
# Let's run the same snippet from above with a monitor in place and take a look at how queuing latency evolves over time. If requests spend longer and longer queuing over time, then our network is acting as a bottleneck and we need to scale it up. Otherwise, something else is going wrong.

# In[11]:


# set up a new directory just for our metrics
metrics_dir = Path("metrics")
metrics_dir.mkdir(exist_ok=True)
metrics_file = metrics_dir / "non-streaming_single-model.csv"

callback = Callback(num_inferences)
with serve("model-repo", gpus=[0]) as instance:
    logger.info("Waiting for inference service to come online...")
    instance.wait()
    logger.info("Service ready!")

    client = InferenceClient(
        "localhost:8001",
        model_name="my-classifier",
        model_version=1,
        callback=callback
    )
    monitor = ServerMonitor(
        model_name="my-classifier",
        ips="localhost",
        filename=metrics_file,
        model_version=1,
        name="monitor",
        rate=4
    )

    with client, monitor:
        for i in range(int(num_inferences)):
            start = i * inference_stride
            stop = start + kernel_size
            kernel = hoft[:, start: stop][None]

            # for various reasons, the rate limiter won't work
            # here, so we'll be generous and insert a sleep
            # for half as long as the actual rate would be
            client.infer(kernel, request_id=i)
            time.sleep(0.5 / INFERENCE_SAMPLING_RATE / INFERENCE_RATE)

        while True:
            results = client.get()
            if results is not None:
                logger.info("Inference complete!")
                break


# Now let's load in the CSV that our monitor produced and take a look at the data it captured.

# In[12]:


df = pd.read_csv(metrics_file)
df


# The `queue`, `compute_input`, `compute_infer`, and `compute_output` columns all represent different steps in the inference compute of a single request. The values in each column represent the cumulative microseconds spent on each step over all of the requests computed between each ping to the metrics service, the number of which is indicated by the `count` column. Knowing this, let's reframe some of the info in these columns in a more useful fashion for our purposes.

# In[13]:


# we'll be doing this a lot, so let's record it as a function
def cleanup_df(df, t0):
    df["Time since start (s)"] = df["timestamp"] - t0
    df["Average queue time (us)"] = df["queue"] / df["count"]

    # count all inference steps as a single metric
    # of inference latency
    infer_time = df[[f"compute_{i}" for i in ["input", "infer", "output"]]].sum(axis=1)
    df["Average infer time (us)"] = infer_time / df["count"]

    # use the number of inferences completed in an interval
    # along with the inference sampling rate to put throughput
    # in units of data seconds per second
    df["Throughput (s' / s)"] = df["count"] / df["timestamp"].diff() / INFERENCE_SAMPLING_RATE

    return df[[
        "Time since start (s)",
        "Throughput (s' / s)",
        "Average queue time (us)",
        "Average infer time (us)"
    ]]

df = cleanup_df(df, df.timestamp.min())
df


# Now let's plot the queue time and throughput as a function of time, and consider what this data tells us:

# In[14]:


p = plotting.plot_inference_metrics_vs_time(my_classifer=df)
plotting.show(p)


# Counterintuitively, our queue time starts off really high, then goes *down* over time! Moreover, the length of the x-axis, around 3-4 seconds, doesn't match up with the 10 second time delta we see between our logs above. What's going on here?
# 
# It turns out that the first couple of inferences can often take substantially longer than the rest, as Triton tries to optimize the compute kernels used to actually perform inference. So while the first request is taking several seconds to process, an enormous queue builds up while we inundate the server with follow-ups. Meanwhile, the `ServerMonitor` doesn't start saving metrics until the server-side metrics service indicates that at least one inference has completed, which explains why we see a much shorter observation window.
# 
# Once that first request is processed, our throughput skyrockets while the network tears through data as fast as it can (in fact this gives a decent estimate of the network's individual throughput cap: around 1200 s'/s). However, we aren't supplying _new_ data to the network fast enough to keep it saturated, so the queue time and throughput both eventually settle down to some roughly constant values. Unlike the first request issue, which can be solved trivially by introducing a block until the first response has been received, _this_ is a real problem. If we can't get data to Triton fast enough now, with a relatively simple set up, then we'll get no benefit from the myriad ways Triton, and inference-as-a-service more broadly, make scaling up easy.
# 
# So _why_ is this issue occurring? Well, think about what happened when we went from `INFERENCE_SAMPLING_RATE = 0.25` to `INFERENCE_SAMPLING_RATE = 4`: we're continuing to send `KERNEL_LENGTH`-long windows of data to the inference service with each request, but now we're doing it 16 times as often per second, with largely redundant data! It's this network I/O that's bottlenecking our pipeline now.
# 
# To alleviate this, we'll need to build a model on the server-side which can cache data we've already sent, and use it to along with updates of *new* data to build the windows we need and pass them along to our downstream model. `hermes.quiver` has built-in support for constructing such a **model ensemble** by adding in a "snapshotter" model on the front-end of the server which maintains the state of the most recent input snapshot.

# In[15]:


# add a new meta-model to the repository that organizes
# graphs of existing models to pass outputs from one
# as inputs to the next
ensemble = repo.add("streaming-classifier", platform=qv.Platform.ENSEMBLE)

# insert a snapshotter model at the front of this ensemble
# whose output will be passed to the input of my-classifier
classifier = repo.models["my-classifier"]
ensemble.add_streaming_inputs(
    classifier.inputs["hoft"],
    stream_size=inference_stride,
    batch_size=1
)

# we know our first request takes ~12s, so make
# sure that the snapshotter will maintain a state
# for longer than this
snapshotter = repo.models["snapshotter"]
snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(25 * 10**6)
snapshotter.config.write()

# mark the output of our classifier as the output
# of the whole ensemble and then export a "version"
# of the ensemble (basically just writes its config)
ensemble.add_output(classifier.outputs["prob"])
ensemble.export_version(None)


# Now we can run inference on our streaming model and only send the updates we need for each request, rather than the entire window!

# In[16]:


# keep our inference results from earlier to
# verify that this implementation comes out the same
nonstreaming_results = results

# quick addition to the callback to handle waiting
# for the first couple responses to come back
class BlockingCallback(Callback):
    def block(self, i):
        while self.y[i] == 0:
            time.sleep(1e-3)

# we need to do more inferences this time, since
# the snapshot state gets initialized to 0s, so the
# first KERNEL_LENGTH * INFERENCE_SAMPLING_RATE updates
# just function to fill the snapshot out
num_inferences = inference_data_size // inference_stride
callback = BlockingCallback(num_inferences)

metrics_file = metrics_dir / "streaming_single-model.csv"
with serve("model-repo", gpus=[0]) as instance:
    logger.info("Waiting for inference service to come online...")
    instance.wait()
    logger.info("Service ready!")

    client = InferenceClient(
        "localhost:8001",
        model_name="streaming-classifier",
        model_version=1,
        callback=callback
    )
    monitor = ServerMonitor(
        model_name="streaming-classifier",
        ips="localhost",
        filename=metrics_file,
        model_version=1,
        name="monitor"
    )

    with client, monitor:
        for i in range(int(num_inferences)):
            start = i * inference_stride
            stop = start + inference_stride  # note the smaller slice
            kernel = hoft[:, start: stop]

            # provide some additional information to
            # the inference server to allow us to keep
            # track of multiple different streams
            with rate_limiter:
                client.infer(
                    kernel,
                    request_id=i,
                    sequence_id=1001,
                    sequence_start=i == 0,
                    sequence_end=(i + 1) == num_inferences
                )

            if i < 3:
                callback.block(i)
            if i == 2:
                logger.info("First 3 requests completed")

        while True:
            results = client.get()
            if results is not None:
                logger.info("Inference complete!")
                break

# ditch some of the initial inferences, which took
# place on a 0-initialized kernel with some updates
# placed at the end of it
results = results[int(KERNEL_LENGTH * INFERENCE_SAMPLING_RATE) - 1:]

# validate that this implementation produces the
# same resluts as the original
assert (results == nonstreaming_results).all()


# In[17]:


p = plotting.plot_timeseries(results, INFERENCE_SAMPLING_RATE, KERNEL_LENGTH)
plotting.show(p)


# In[18]:


df = pd.read_csv(metrics_file)
dfs = {}
for model, subdf in df.groupby("model"):
    subdf = cleanup_df(subdf.reset_index(), df.timestamp.min())
    dfs[model] = subdf

p = plotting.plot_inference_metrics_vs_time(**dfs)
plotting.show(p)


# So this looks like a pretty stable configuration, which means we can up our inference rate until we start to hit a bottleneck. Let's hop up to 400 and see what happens.

# In[19]:


INFERENCE_RATE = 400
kernels_per_second = int(INFERENCE_RATE * INFERENCE_SAMPLING_RATE)
rate_limiter = RateLimiter(max_calls=kernels_per_second / 20, period=0.05)
callback = BlockingCallback(num_inferences)

metrics_file = metrics_dir / "streaming_rate-400_single-model.csv"
with serve("model-repo", gpus=[0]) as instance:
    logger.info("Waiting for inference service to come online...")
    instance.wait()
    logger.info("Service ready!")

    client = InferenceClient(
        "localhost:8001",
        model_name="streaming-classifier",
        model_version=1,
        callback=callback
    )
    monitor = ServerMonitor(
        model_name="streaming-classifier",
        ips="localhost",
        filename=metrics_file,
        model_version=1,
        name="monitor",
        rate=4,
    )

    with client, monitor:
        for i in range(int(num_inferences)):
            start = i * inference_stride
            stop = start + inference_stride
            kernel = hoft[:, start: stop]

            with rate_limiter:
                client.infer(
                    kernel,
                    request_id=i,
                    sequence_id=1001,
                    sequence_start=i == 0,
                    sequence_end=(i + 1) == num_inferences
                )

            if i < 3:
                callback.block(i)
            if i == 2:
                logger.info("First 3 requests completed")

        while True:
            results = client.get()
            if results is not None:
                logger.info("Inference complete!")
                break


# In[20]:


df = pd.read_csv(metrics_file)
dfs = {}
for model, subdf in df.groupby("model"):
    subdf = cleanup_df(subdf.reset_index(), df.timestamp.min())
    dfs[model] = subdf

p = plotting.plot_inference_metrics_vs_time(**dfs)
plotting.show(p)


# So it looks like our queue latencies are still reasonably stable, but our throughput ends up tanking early in our inference run. The likely culprit is that we've saturated the rate at which our client can generate requests: gRPC's asynchronous requests are implemented with Python threads in a way that doesn't lend itself to doing strenuous work in the main thread which generates the requests. We can improve throughput further by sending longer updates at a lower rate, then using the caching model on the server to build *batches* of kernels to send to the downstream model.
# 
# Let's export a new ensemble and associated snapshotter model that expects batched updates, then run inference with that.

# In[21]:


batched_ensemble = repo.add("batched-streaming-classifier", platform=qv.Platform.ENSEMBLE)

# right now we can only handle batches small
# enough that the update isn't longer than
# the kernel itself, so we'll use the biggest
# batch we can with that constraint
batch_size = int(KERNEL_LENGTH * INFERENCE_SAMPLING_RATE)
classifier = repo.models["my-classifier"]
batched_ensemble.add_streaming_inputs(
    classifier.inputs["hoft"],
    stream_size=inference_stride,
    batch_size=batch_size,
    name="batched-snapshotter"
)

# we know our first request takes ~12s, so make
# sure that the snapshotter will maintain a state
# for longer than this
snapshotter = repo.models["batched-snapshotter"]
snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(25 * 10**6)
snapshotter.config.write()

# mark the output of our classifier as the output
# of the whole ensemble and then export a "version"
# of the ensemble (basically just writes its config)
batched_ensemble.add_output(classifier.outputs["prob"])
batched_ensemble.export_version(None)


# In[22]:


# make a new callback that's better equipped to
# slice out a batch of responses
class BatchedCallback(BlockingCallback):
    def __init__(self, num_inferences, batch_size):
        self.y = np.zeros((num_inferences * batch_size,))
        self.batch_size = batch_size

    def __call__(self, response, request_id, sequence_id):
        start = request_id * self.batch_size
        self.y[start: start + len(response)] = response[:, 0]
        if (start + len(response) + 1) >= len(self.y):
            return self.y


num_kernels = inference_data_size // inference_stride
num_inferences = num_kernels // batch_size
callback = BatchedCallback(num_inferences, batch_size)

batches_per_second = int(INFERENCE_RATE * INFERENCE_SAMPLING_RATE / batch_size)
rate_limiter = RateLimiter(max_calls=batches_per_second / 20, period=0.05)

metrics_file = metrics_dir / "batched-streaming_single-model.csv"

with serve("model-repo", gpus=[0]) as instance:
    logger.info("Waiting for inference service to come online...")
    instance.wait()
    logger.info("Service ready!")

    client = InferenceClient(
        "localhost:8001",
        model_name="batched-streaming-classifier",
        model_version=1,
        callback=callback
    )
    monitor = ServerMonitor(
        model_name="batched-streaming-classifier",
        ips="localhost",
        filename=metrics_file,
        model_version=1,
        name="monitor"
    )

    with client, monitor:
        for i in range(int(num_inferences)):
            start = i * inference_stride * batch_size
            stop = (i + 1) * inference_stride * batch_size
            kernel = hoft[:, start: stop]

            with rate_limiter:
                client.infer(
                    kernel,
                    request_id=i,
                    sequence_id=1001,
                    sequence_start=i == 0,
                    sequence_end=(i + 1) == num_inferences
                )

            if i < 3:
                callback.block(i)
            if i == 2:
                logger.info("First 3 requests completed")

        while True:
            results = client.get()
            if results is not None:
                logger.info("Inference complete!")
                break


# In[23]:


df = pd.read_csv(metrics_file)
dfs = {}
for model, subdf in df.groupby("model"):
    subdf = cleanup_df(subdf.reset_index(), df.timestamp.min())
    subdf["Throughput (s' / s)"] *= batch_size
    dfs[model] = subdf

p = plotting.plot_inference_metrics_vs_time(**dfs)
plotting.show(p)


# Ok, so we're back to a stable system. Let's try pushing this a bit farther, and use a bit longer of a segment to be able to get a decent estimate of our metrics.

# In[24]:


INFERENCE_DATA_LENGTH = 32768
INFERENCE_RATE = 4096

inference_data_size = INFERENCE_DATA_LENGTH * SAMPLE_RATE
hoft = np.random.randn(NUM_IFOS, inference_data_size).astype("float32")

num_kernels = inference_data_size // inference_stride
num_inferences = num_kernels // batch_size
callback = BatchedCallback(num_inferences, batch_size)

batches_per_second = int(INFERENCE_RATE * INFERENCE_SAMPLING_RATE / batch_size)
rate_limiter = RateLimiter(max_calls=batches_per_second / 20, period=0.05)

metrics_file = metrics_dir / "batched-streaming_rate-4096_single-model.csv"

with serve("model-repo", gpus=[0]) as instance:
    logger.info("Waiting for inference service to come online...")
    instance.wait()
    logger.info("Service ready!")

    client = InferenceClient(
        "localhost:8001",
        model_name="batched-streaming-classifier",
        model_version=1,
        callback=callback
    )
    monitor = ServerMonitor(
        model_name="batched-streaming-classifier",
        ips="localhost",
        filename=metrics_file,
        model_version=1,
        name="monitor"
    )

    with client, monitor:
        for i in range(int(num_inferences)):
            start = i * inference_stride * batch_size
            stop = (i + 1) * inference_stride * batch_size
            kernel = hoft[:, start: stop]

            with rate_limiter:
                client.infer(
                    kernel,
                    request_id=i,
                    sequence_id=1001,
                    sequence_start=i == 0,
                    sequence_end=(i + 1) == num_inferences
                )

            if i < 3:
                callback.block(i)
            if i == 2:
                logger.info("First 3 requests completed")

        while True:
            results = client.get()
            if results is not None:
                logger.info("Inference complete!")
                break


# In[25]:


df = pd.read_csv(metrics_file)
dfs = {}
for model, subdf in df.groupby("model"):
    subdf = cleanup_df(subdf.reset_index(), df.timestamp.min())
    subdf["Throughput (s' / s)"] *= batch_size
    dfs[model] = subdf

p = plotting.plot_inference_metrics_vs_time(**dfs)
plotting.show(p)

