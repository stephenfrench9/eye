#!/bin/bash -e
# You need to install the AWS Command Line Interface from http://aws.amazon.com/cli/
AMIID="$(aws ec2 describe-images --filters "Name=name,Values=amzn-ami-hvm-2017.09.1.*-x86_64-gp2" --query "Images[0].ImageId" --output text)"
AMIID="ami-0a8686e2631bb688d"
VPCID="$(aws ec2 describe-vpcs --filter "Name=isDefault, Values=true" --query "Vpcs[0].VpcId" --output text)"
SUBNETID="$(aws ec2 describe-subnets --filters "Name=vpc-id, Values=$VPCID" --query "Subnets[0].SubnetId" --output text)"
SGID="$(aws ec2 create-security-group --group-name mysecuritygroup$1 --description "My security group" --vpc-id "$VPCID" --output text)"
aws ec2 authorize-security-group-ingress --group-id "$SGID" --protocol tcp --port 22 --cidr 0.0.0.0/0
INSTANCEID="$(aws ec2 run-instances --image-id "$AMIID" --key-name awsKey --instance-type g3.4xlarge --security-group-ids "$SGID" --subnet-id "$SUBNETID" --query "Instances[0].InstanceId" --output text)"
echo "waiting for $INSTANCEID ..."
aws ec2 wait instance-running --instance-ids "$INSTANCEID"

PUBLICNAME="$(aws ec2 describe-instances --instance-ids "$INSTANCEID" --query "Reservations[0].Instances[0].PublicDnsName" --output text)"

echo "$INSTANCEID" >> instanceid.txt
echo "$PUBLICNAME" >> publicname.txt

#'bash -s' < testscrip.sh
#scp -r -i "awsKey.pem" ec2-user@$PUBLICNAME:/home/ec2-user/hygens/ .
echo "$INSTANCEID is accepting SSH connections under $PUBLICNAME"
echo "ssh -i awsKey.pem ec2-user@$PUBLICNAME"
#ssh -i "awsKey.pem" ec2-user@$PUBLICNAME

#read -r -p "Press [Enter] key to terminate $INSTANCEID ..."
#aws ec2 terminate-instances --instance-ids "$INSTANCEID"
#echo "terminating $INSTANCEID ..."
#aws ec2 wait instance-terminated --instance-ids "$INSTANCEID"
#aws ec2 delete-security-group --group-id "$SGID"
#echo "done."