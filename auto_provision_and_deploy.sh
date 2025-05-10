#!/bin/bash
set -e

echo "🚀 [1/5] Initializing Terraform..."
cd tf/
terraform init

echo "📦 [2/5] Applying Terraform configuration..."
terraform apply -auto-approve -var="nodes={\"node1\":\"training\",\"node2\":\"serving\"}"

echo "🌐 [3/5] Extracting floating IPs..."
TRAINING_IP=$(terraform output -raw training_floating_ip)
SERVING_IP=$(terraform output -raw serving_floating_ip)

echo "🔧 [4/5] Generating dynamic Ansible inventory..."
cd ../ansible/
cat <<EOF > inventory.yml
all:
  vars:
    ansible_user: cc
    ansible_ssh_private_key_file: ~/.ssh/project_24.pem

  hosts:
    training_node:
      ansible_host: ${TRAINING_IP}
    serving_node:
      ansible_host: ${SERVING_IP}
EOF

echo "🚀 [5/5] Running Ansible deployment playbook..."
ansible-playbook deploy.yml -i inventory.yml

echo "✅ Deployment complete!"
