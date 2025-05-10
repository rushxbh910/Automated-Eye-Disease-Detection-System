variable "suffix" {
  description = "Unique name suffix, e.g., your NetID"
  type        = string
}

variable "key" {
  description = "Name of the SSH keypair registered with Chameleon"
  type        = string
}

variable "flavor" {
  description = "Instance flavor (CPU/mem combo)"
  default     = "m1.medium"
  type        = string
}

variable "nodes" {
  default = {
    node1 = {}
    node2 = {}
    node3 = {}
  }
}
