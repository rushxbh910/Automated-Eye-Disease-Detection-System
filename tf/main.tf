resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes
  name            = "eye-node-${each.key}-${var.suffix}"
  flavor_name     = var.flavor
  key_pair        = var.key
  image_id        = data.openstack_images_image_v2.ubuntu.id

  network {
    port = openstack_networking_port_v2.sharednet1_ports[each.key].id
  }

  security_groups = ["default", "eye-secgroup-${var.suffix}"]
}

resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each    = var.nodes
  name        = "eye-port-${each.key}-${var.suffix}"
  network_id  = data.openstack_networking_network_v2.sharednet1.id

  fixed_ip {
    subnet_id = data.openstack_networking_subnet_v2.sharednet1_subnet.id
  }

  security_group_ids = [
    data.openstack_networking_secgroup_v2.default.id,
    openstack_networking_secgroup_v2.eye_secgroup.id
  ]
}

resource "openstack_networking_floatingip_v2" "fip" {
  pool = "public"
}

resource "openstack_networking_floatingip_associate_v2" "fip_assoc" {
  floating_ip = openstack_networking_floatingip_v2.fip.address
  port_id     = openstack_networking_port_v2.sharednet1_ports["node1"].id
}

resource "openstack_networking_secgroup_v2" "eye_secgroup" {
  name = "eye-secgroup-${var.suffix}"
}

resource "openstack_networking_secgroup_rule_v2" "inbound" {
  for_each = toset(["22", "80", "443", "8080", "8081", "8888", "9000", "9001"])
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = each.value
  port_range_max    = each.value
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.eye_secgroup.id
}
