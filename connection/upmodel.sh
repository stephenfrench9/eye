#!/usr/bin/env bash
scp -r -i awsKey.pem ~/aaa/models/9-10-5/ ec2-user@$(cat publicname.txt):/home/ec2-user/eye/models/