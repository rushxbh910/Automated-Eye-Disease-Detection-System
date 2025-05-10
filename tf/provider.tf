provider "openstack" {
  cloud = "chi"
}

provider "openstack" {
  cloud = "kvm"
  alias = "kvm"
}
