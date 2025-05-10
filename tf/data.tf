data "openstack_images_image_v2" "ubuntu" {
  name = "CC-Ubuntu22.04"
}

data "openstack_networking_network_v2" "sharednet1" {
  name = "sharednet1"
}

data "openstack_networking_subnet_v2" "sharednet1_subnet" {
  name = "sharednet1-subnet"
}

data "openstack_networking_secgroup_v2" "default" {
  name = "default"
}
