# Installation

The library was developed using the Anaconda3 distribution of Python 3.11.5. In the following instructions for the installation via conda as well as pip are provided.

## Prerequisites
- A working GitHub installation.
- [ArraySoft v2.30](https://cdn.website-editor.net/s/156d2965ff764637aaea150903bb0161/files/uploaded/SetupHTPAdGUIv2_30.rar?Expires=1709501376&Signature=p8nizQ9W1PC3O4uuuEpsxpqhTZc3t1vdlI1HImzhGcxrSur-9jmvYcf7EvJvU223HmZKhFJvr4dYW8PYtFwv1RVGjh626sN0ZQRICL6MBwOhqmevGODlUCFYjuGCMGWwlJCMpVz68dIYcBKjBS7MhEGKL~wCf1atkW82yr6eewPK3AJQmV0StLWQCi7Z4Q8epYWjGt4Xmuaa7wAcQJfFCBK1IKutkl52FPint4CqYarqQKfqpKJMn13SlVQSdp-RZbKZirddGrQkzTpYfi2BOuxThTn6C-FfXGQLC~Hnt1858gv96EKq23VJEGDmV~97rhQmj2YzXGySGpMnVv5xJQ__&Key-Pair-Id=K2NXBXLF010TJW) (if .bds or .txt files are to be parsed to pandas)

## Download HSPyTools

1. Clone this repository either via using the shell
   ```sh
   git clone https://github.com/arehmer/HSPyTools.git
    ```
   or using the GitHub Desktop-App: ```File-->Clone repository ``` <br>
   <img src="images/screenshot1_github_desktop.png"  height="300">

   and then enter the URL ```https://github.com/arehmer/HSFit.git``` into the mask <br>
   <img src="images/screenshot2_github_desktop.png" height="200">

## Install HSPyTools
1. Optional: Activate the environment in which HSPyTools is to be installed.
2. Install HSPyTools in development mode with pip via
    ```sh
    pip install -e \path\to\HSPyTools>
    ```
## Check installation
1. In the conda shell type
   ```sh
   python
   ```
2. Then try to import HSPyTools by typing
   ```sh
   import hspytools
   ```
