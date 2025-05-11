variable "key" {}
variable "suffix" {}
variable "nodes" {
  type = map(object({
    flavor = string
  }))
}
