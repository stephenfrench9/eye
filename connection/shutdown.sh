IID="$(cat instanceid.txt)"
aws ec2 terminate-instances --instance-ids "$IID"