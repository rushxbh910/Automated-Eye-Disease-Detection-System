import boto3

minio_client = Minio(
    endpoint="minio:9000",  # NOT localhost!
    access_key="your-access-key",
    secret_key="your-secret-key",
    secure=False
)


bucket_name = "production"
image_path = "sample.jpeg"
object_key = "feedback_samples/sample.jpeg"

with open(image_path, "rb") as f:
    minio_client.upload_fileobj(f, bucket_name, object_key)

print(f"✅ Uploaded '{object_key}' to bucket '{bucket_name}'")