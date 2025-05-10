variable "suffix" {
  description = "Project or user suffix for resource naming"
  type        = string
}

variable "key" {
  description = "SSH key pair name"
  type        = string
}

variable "nodes" {
  description = "Map of training node(s) and their configuration"
  type = map(object({
    role   = string
    flavor = string
  }))
}

variable "model_serving_flavor" {
  description = "Flavor name for model serving node"
  type        = string
  default     = "m1.large"
}