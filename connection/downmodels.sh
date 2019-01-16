#!/usr/bin/env bash
scp -r -i awsKey.pem ec2-user@$(cat publicname.txt):/home/ec2-user/eye/models/ .