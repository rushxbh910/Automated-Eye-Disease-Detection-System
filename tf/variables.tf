variable "key" {
  type = string
  description = "SSH key pair name"
}

variable "suffix" {
  type = string
  description = "Unique name suffix"
}

variable "nodes" {
  description = "Map of nodes with their roles and flavors"
  type = map(object({
    flavor = string
  }))
}