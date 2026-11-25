# Deploy the DurLAR viewer on Railway Hobby

The deployment package contains the app and calibration files. It downloads the July 16, 2021 drive to a 5 GB volume after the service starts. That drive has 600 paired frames and uses about 3.4 GiB.

## Create the service

Run the commands in the extracted `durlar-railway` directory. If the Railway CLI asks you to sign in, run `railway login` first.

```sh
railway init --name durlar-viewer
railway add --service durlar-viewer
railway service link durlar-viewer
railway volume add --mount-path /app/data
railway up --ci
```

These commands create a new project. For a retry in the same project, resume at the failed command. Set the service's health check path to `/health` if you enable health checks in the Railway dashboard.

## Load the drive and open the viewer

After Railway shows the service as running, run:

```sh
railway ssh -- python3 /app/download_durlar.py --local-dir /app/data --drive DurLAR_20210716_S --all-frames
railway domain
```

If Railway asks for an SSH key, follow its setup prompt. The downloader skips files already present, so you can rerun the SSH command after an interruption. After it finishes, open the URL from `railway domain` and select `DurLAR_20210716_S`. The generated domain is public.

The deployment serves data in read-only mode. By default, it streams one in four LiDAR returns, about 1 MiB per recorded frame before the camera image. Use the Sequence menu to choose the drive and the timeline to step through frames.

[Railway CLI deployment](https://docs.railway.com/cli/deploying) · [Railway volumes](https://docs.railway.com/volumes) · [DurLAR_S dataset](https://huggingface.co/datasets/l1997i/DurLAR_S)
