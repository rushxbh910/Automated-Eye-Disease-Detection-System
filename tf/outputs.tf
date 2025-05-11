output "training_node_name" {
  value = openstack_compute_instance_v2.training_node.name
}

output "serving_node_name" {
  value = openstack_compute_instance_v2.serving_node.name
}