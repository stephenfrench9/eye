#!/bin/bash
cd eye/
git pull origin master
sudo docker run -itv $(pwd):/ralston kaggle/python
