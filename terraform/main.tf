provider "chameleon" {
  endpoint = var.chameleon_endpoint
  region   = var.region
}

resource "chameleon_volume" "data" {
  name = "eye-data-block-project${var.project_id}"
  size = 200
}

resource "chameleon_instance" "training" {
  name     = "eye-train-project${var.project_id}"
  flavor   = "gpu_a100_pcie"
  image    = "ubuntu-22.04"
  key_pair = var.ssh_key
  volumes  = [chameleon_volume.data.id]
}

resource "chameleon_instance" "serve" {
  name     = "eye-serve-project${var.project_id}"
  flavor   = "gpu_v100"
  image    = "ubuntu-22.04"
  key_pair = var.ssh_key
}

resource "chameleon_instance" "monitor" {
  name     = "eye-monitor-project${var.project_id}"
  flavor   = "m1.medium"
  image    = "ubuntu-22.04"
  key_pair = var.ssh_key
}

resource "chameleon_instance" "ray_head" {
  name     = "ray-head-project${var.project_id}"
  flavor   = "compute_liqid"
  image    = "ubuntu-22.04"
  key_pair = var.ssh_key
}

resource "chameleon_instance" "ray_worker" {
  count    = 2
  name     = "ray-worker-${count.index + 1}-project${var.project_id}"
  flavor   = "compute_liqid"
  image    = "ubuntu-22.04"
  key_pair = var.ssh_key
}

resource "chameleon_instance" "runner" {
  name     = "runner-project${var.project_id}"
  flavor   = "m1.small"
  image    = "ubuntu-22.04"
  key_pair = var.ssh_key
}

output "training_ip"    { value = chameleon_instance.training.access_ip_v4 }
output "serve_ip"       { value = chameleon_instance.serve.access_ip_v4 }
output "monitor_ip"     { value = chameleon_instance.monitor.access_ip_v4 }
output "ray_head_ip"    { value = chameleon_instance.ray_head.access_ip_v4 }
output "ray_worker_ips"{ value = [for w in chameleon_instance.ray_worker : w.access_ip_v4] }
output "runner_ip"      { value = chameleon_instance.runner.access_ip_v4 }
