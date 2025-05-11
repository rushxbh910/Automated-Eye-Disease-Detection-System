variable "suffix" {
  type        = string
  description = "Project/user suffix"
}

variable "key" {
  type        = string
  description = "SSH key name"
}

variable "nodes" {
  type = map(object({
    flavor = string
  }))
}
