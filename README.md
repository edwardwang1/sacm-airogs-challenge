# AIROGS Challenge Algorithm
This is the github repository for the Schulich Applied Computing in Medicine's submission to the [AIROGS challenge](https://airogs.grand-challenge.org/). It builds a Docker Container that works with the [Grand Challenge](https://grand-challenge.org/blogs/create-an-algorithm/) website.

## Modifying pretrained networks
A dropout layer was inserted before the last fully connected layer in all networks used. As an example, this is how it is done for Densenet. The full code for this is available in process.py. If you want to load the trained weights outside of this docker container, you will need to modify the pretrained model appropriately. 

    model = models.densenet161(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 1, bias=True))
    model.load_state_dict(torch.load(DENSENET_WEIGHT_PATH))

## Building on Windows
If you want to use this container on Windows, you will need to have [Docker](https://docs.docker.com/) installed on your system with [WSL 2.0](https://docs.microsoft.com/en-us/windows/wsl/install). It should be possible to use [CUDA with WSL](https://developer.nvidia.com/cuda/wsl), but this was not tested. If no GPU is detected, Pytorch will use CPU.

## Competition Paper
The competition paper will be uploaded when the challenge ends.


