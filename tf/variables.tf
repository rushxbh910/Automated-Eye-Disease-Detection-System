
variable "chameleon_endpoint" {
  default = "https://chi.tacc.chameleoncloud.org:5000/v3"
}

variable "region" {
  default = "CHI@TACC"
}

variable "key" {
  description = "SSH Key pair name"
  type        = string
  default     = "project_24"
}

variable "suffix" {
  description = "Unique identifier (e.g. your ProjectID)"
  default     = "project_24"
}

variable "nodes" {
  description = "Map of training nodes"
  type = map(object({
    flavor = string
  }))
}

variable "model_serving_flavor" {
  default     = "m1.large"
  description = "Flavor for serving node on KVM"
}
