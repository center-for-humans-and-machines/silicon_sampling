# Handling Cluster Queues Correctly
Depending on the Cluster that you work on, the jobs that you submit might be stuck in queue for a long time.

Here I'll share some do's and don't's regarding cluster management and share some information on how the clusters scheduling work.

### Do's
* Set time limits with moderate headroom regarding the expected runtime for you job (maybe 50%).
* Combine multiple tasks into one sequential or parallel job (Especially if they involve model loading).
* Set checks so you notice when runs are broken early.
* Immediately cancel broken runs to not waste resources.

### Don't
* Run many individual short pilot jobs. (Each of them may wait in queue for a long time)
* Split tasks up into multiple cluster jobs. (Each of them may wait in queue for a long time)
* Just set time limits to 24h no matter the expected runtime. (May increase queue time)
* Cancel and resubmit a job with different parameters just because it is "stuck in queue" (It won't help (and will hurt). The cluster is just congested)

## DAIS specific information
DAIS can be very much differently congested. A job might start instantly or take more than 24h to start. Because we're using DAIS heavily, our "fairshare" is fairly low, so many other jobs automatically get higher priority.

The primary driver of queue wait time on DAIS is number of GPUs / Nodes requested. Secondary driver is time limit.

For short jobs that don't strictly require many GPUs, it is typically smarter to go for a smaller job to avoid the queue time.

I.e. if not otherwise specified, it is smarter to request 1 GPU for 4 hours than 4 GPU for 1 hour.

DAIS squeue gives various status information codes for why a job didn't start yet:

1. Priority -> A job with higher priority is queued for the same resources. It will start when nodes free before this job.
2. Resources -> The job is queued for starting as soon a new resources free. Status might jump back to "Priority" if a higher priority job is submitted.
3. Nodes required for job are down ... -> Similar to "Priority", usually temporary.

If you see any of these codes, don't worry and just wait for your job to start. If you see any other code and the job isn't starting, please inform the user.

### DAIS GPUs
DAIS has 10 8xB200 and 17 8xH200 Nodes. By default, your jobs will request GPU type. But depending on requirements, use gpu_type to set a GPU type. You may run sinfo on the cluster to figure out how busy H200 and B200 nodes are. H200 nodes are daisg1.. nodes, B200 are daisg2.. nodes.