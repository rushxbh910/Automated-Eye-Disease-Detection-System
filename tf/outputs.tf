output "floating_ip" {
  value = openstack_networking_floatingip_v2.fip.address
}

output "node1_name" {
  value = openstack_compute_instance_v2.nodes["node1"].name
}
