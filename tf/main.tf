locals {
  training_node = var.nodes["train-1"]
  serving_node  = var.nodes["serve-1"]
}

# --------------------
# CHI@TACC: Training Node (GPU Bare Metal via Reservation - No Ports, No Security Groups)
# --------------------
resource "openstack_compute_instance_v2" "training_node" {
  provider     = openstack.chi
  name         = "eye-train-train-1-${var.suffix}"
  flavor_name  = local.training_node.flavor
  key_pair     = var.key
  image_id     = data.openstack_images_image_v2.ubuntu.id
  reservation_id = var.training_reservation_id

  network {
    name = "sharednet1"
  }
}

# --------------------
# KVM@TACC: Serving Node (VM with Security Groups and Floating IP)
# --------------------
resource "openstack_compute_instance_v2" "serving_node" {
  provider        = openstack.kvm
  name            = "eye-serve-serve-1-${var.suffix}"
  flavor_name     = local.serving_node.flavor
  key_pair        = var.key
  image_id        = data.openstack_images_image_v2.ubuntu.id

  network {
    port = openstack_networking_port_v2.serving_port.id
  }

  security_groups = ["default", "eye-secgroup-${var.suffix}"]
}

resource "openstack_networking_port_v2" "serving_port" {
  provider   = openstack.kvm
  name       = "eye-serve-port-serve-1-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet1.id

  fixed_ip {
    subnet_id = data.openstack_networking_subnet_v2.sharednet1_subnet.id
  }

  security_group_ids = [
    data.openstack_networking_secgroup_v2.default.id,
    openstack_networking_secgroup_v2.eye_secgroup.id
  ]
}

resource "openstack_networking_floatingip_v2" "serving_fip" {
  provider = openstack.kvm
  pool     = "public"
}

resource "openstack_networking_floatingip_associate_v2" "serving_fip_assoc" {
  provider    = openstack.kvm
  floating_ip = openstack_networking_floatingip_v2.serving_fip.address
  port_id     = openstack_networking_port_v2.serving_port.id
}

# --------------------
# Security Group for Serving Node
# --------------------
resource "openstack_networking_secgroup_v2" "eye_secgroup" {
  provider = openstack.kvm
  name     = "eye-secgroup-${var.suffix}"
}

resource "openstack_networking_secgroup_rule_v2" "inbound" {
  provider         = openstack.kvm
  for_each         = toset(["22", "80", "443", "8080", "8081", "8888", "9000", "9001"])
  direction        = "ingress"
  ethertype        = "IPv4"
  protocol         = "tcp"
  port_range_min   = each.value
  port_range_max   = each.value
  remote_ip_prefix = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.eye_secgroup.id
}