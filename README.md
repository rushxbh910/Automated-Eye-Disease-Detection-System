
## Automated Eye Disease Detection System for Enhanced Clinical Diagnostics

#### Value Proposition:
<!-- Our project focuses on developing a machine learning system that seamlessly integrates into the existing diagnostic framework within eye hospitals and clinics. We're not trying to create a new business; instead, we're aiming to augment the capabilities of current medical practices. This system will analyze retinal images, providing ophthalmologists with an automated tool to aid in the early and accurate detection of a range of eye diseases, including Retinitis Pigmentosa, Retinal Detachment, Pterygium, Myopia, Macular Scar, Glaucoma, Disc Edema, Diabetic Retinopathy, and Central Serous Chorioretinopathy. -->
Our project focuses on developing a machine learning system that seamlessly integrates into the existing diagnostic workflow of private ophthalmology practices and small-scale eye clinics. Rather than establishing a new business, our goal is to augment and streamline the capabilities of current medical services.

In this setting, the system serves as a decision-support tool for ophthalmologists. It analyzes retinal images and provides automated, high-confidence predictions for a range of eye diseases—such as Retinitis Pigmentosa, Retinal Detachment, Pterygium, Myopia, Macular Scar, Glaucoma, Disc Edema, Diabetic Retinopathy, and Central Serous Chorioretinopathy.

A dedicated section of the clinic can be used for image acquisition, where patients' retinal scans are captured and processed through the system. The model then outputs a list of the most probable diseases along with confidence scores for each prediction. These results assist ophthalmologists in making faster, more informed diagnostic decisions thereby enhancing clinical efficiency while maintaining expert oversight.

#### Current Status Quo:
<!--Currently, the diagnosis of these diseases relies heavily on manual examination by specialists, a process that can be time-consuming and subjective. Even with advanced imaging technologies like fundus photography and OCT, the interpretation of these images demands significant expertise, often leading to delays in diagnosis. Our system aims to streamline this process, providing quick and consistent analysis to support clinicians in making informed decisions. -->
The clinic is a small private eye clinic run by one or two ophthalmologists. On average, they see about 20 to 30 patients each day. 

Right now, there is no system to help with diagnosing diseases. The doctors look at the eye images themselves and make decisions based on what they see and the patient’s medical history. Everything is done manually.

This takes time—usually around 5 to 10 minutes per patient. When the clinic gets busy, doctors may have even less time. There’s no one else to help check the images, and getting a second opinion is hard because the clinic has limited staff. Sometimes, if the diagnosis isn’t clear, patients have to come back for another visit.

Although the clinic stores eye images digitally, there’s no software to help spot problems automatically. The whole process depends on the doctor’s experience and judgment, which can lead to delays or missed early signs of disease.

#### Business Metric:
<!--The success of our system will be judged on its ability to improve the efficiency and accuracy of eye disease diagnosis. To quantify this, we'll focus on several key business metrics. First, we'll measure the reduction in "time to diagnosis," comparing the time taken to reach a diagnosis with our automated system versus the current manual process. Secondly, we'll assess the "diagnostic accuracy" by determining the concordance rate between our system's predictions and the gold standard diagnosis provided by ophthalmologists. We'll also examine the system's "sensitivity and specificity," ensuring it accurately identifies both positive and negative cases. Additionally, we'll track the "throughput" of our system, which refers to the number of images processed per unit time, to gauge its efficiency in a clinical setting. Finally, and perhaps most importantly, we will focus on the "reduction of false negatives". A false negative could have dire consequences for a patient, and reducing these instances is a paramount concern for our system’s success. By focusing on these metrics, we aim to demonstrate the tangible benefits of our machine learning system in improving patient care and clinical workflows.-->
The success of our system depends on how well it improves the efficiency and accuracy of eye disease diagnosis within the setting of a small private eye clinic. These clinics often face challenges like time constraints, limited staff, and the need to process many patients each day. Our system directly improves this status quo by automating part of the diagnostic process and supporting the ophthalmologist with reliable, fast, and consistent results.

We will measure our impact using several key business metrics:

Reduction in Time to Diagnosis: Currently, doctors spend 5–10 minutes per patient analyzing retinal images. With our system, this time can be reduced significantly—allowing for quicker decision-making and freeing up time to see more patients or focus on complex cases.

Improved Diagnostic Accuracy: By comparing the system’s predictions with the ophthalmologist’s final diagnosis, we aim to increase diagnostic confidence and reduce uncertainty, especially in borderline or early-stage cases.

Sensitivity and Specificity: These metrics ensure that the system reliably detects diseases (true positives) and avoids incorrect alarms (false positives), supporting safe and trusted clinical use.

Increased Throughput: Our model processes images rapidly—targeting under 200 milliseconds per image—allowing it to keep pace with the clinic’s daily flow and avoid backlogs.

Reduction in False Negatives: Missing a disease can have serious consequences. Our system helps catch subtle signs that may be overlooked in a fast-paced clinic setting, ensuring that patients receive timely care and follow-up.

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name             | Responsible for                                                                 | Link to their commits in this repo                                                                 |
|------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Rushabh Bhatt    | CI/CD, Infrastructure as Code, Version Control, Proactive Monitoring and Logging | https://github.com/rushxbh910/Automated-Eye-Disease-Detection-System/commits/main?author=rushxbh910        |
| Shruti Bora      |   Train and Re-train, Training Strategies for Large Models, Scheduling Training Jobs  |  https://github.com/rushxbh910/Automated-Eye-Disease-Detection-System/commits/main/?author=sb9880   |
| Aryan Ajmera     |        model serving and metrics monitoring                                                                          |       https://github.com/rushxbh910/Automated-Eye-Disease-Detection-System/commits?author=AryanAjmera18                                                                                               |
| Vaibhav Rouduri  | Persistent Storage, Offline Data, Data Pipelines, Online Data, Interactive Data Dashboard | https://github.com/rushxbh910/Automated-Eye-Disease-Detection-System/commits/main/?author=vaibhavrouduri                                                                                             |


### System diagram

![System Diagram](System-Diagram.png)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|                              | How it was created | Conditions of use |
|------------------------------|--------------------|-------------------|
| Eye Disease Image Dataset    | A total of 5335 images of healthy and affected eye images were collected from Anwara Hamida Eye Hospital in Faridpur and BNS Zahrul Haque Eye Hospital in Faridpur district with the help of the hospital authorities. Then from these original images, a total of 16242 augmented images are produced by using Rotation, Width shifting, Height shifting, Translation, Flipping, and Zooming techniques to increase the number of data.                     |You can share, copy and modify this dataset so long as you give appropriate credit, provide a link to the CC BY license, and indicate if changes were made, but you may not do so in a way that suggests the rights holder has endorsed you or your use of the dataset. Note that further permission may be required for any content within the dataset that is identified as belonging to a third party.                   |
| Reference for the above      | Riadur Rashid, Mohammad ; Sharmin, Shayla ; Khatun, Tania; Hasan, Md Zahid; Shorif Uddin , Mohammad  (2024), “Eye Disease Image Dataset”, Mendeley Data, V1, doi: 10.17632/s9bfhswzjb.1                   |                   |
| Base model 1                 |  Convolutional Neural Networks (CNNs)| Custom CNN: CNN from scratch with multiple convolutional layers, batch normalization, and dropout for regularization.
|| | EfficientNet (B0-B7): This model is optimized for image classification and provides a good balance between accuracy and efficiency.
|||ResNet (ResNet-50): The residual connections help train deep networks effectively.|
| | Vision Transformers (ViTs)| This model captures long-range dependencies and performs well on large datasets.


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

 | Requirement                                             | How many/when                     | Justification                                    |
 |---------------------------------------------------------|-----------------------------------|--------------------------------------------------|
 | gpu_p100_nvlink/ gpu_a100_pcie/ gpu_v100/ gpu_a100_pcie | approximately for 20 hours        | Image Dataset of 4GB needs high processing power |
 | gpu_v100 | 	1-2 GPUs for real-time inference                 |   Ensures low-latency model serving with a target of <200ms per image.        |
 | gpu_p100/ gpu_mi100/ compute_liqid                      | approximately for 45 hours        | Incase 4 core GPU server not available           |
 | Floating IPs                                            | 3 for entire project duration     |                                                  |
 

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->
###### Model Training at Scale
We're building a convolutional neural network (CNN) to classify eye diseases from medical images. Given CNNs' efficiency in image classification, we’ll experiment with different architectures like ResNet and EfficientNet to find the best balance between accuracy and training time. To keep the model up to date with evolving patterns, we’ll periodically retrain it using new production data.

Since large models can be challenging to train on a single low-end GPU, we’ll explore Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP) techniques to enable multi-GPU training. We’ll also analyze how adjusting batch sizes and using gradient accumulation can improve training efficiency. To understand the impact of distributed training, we’ll compare training times across different GPU setups—single vs. multi-GPU—using DDP and FSDP. Our findings, including a plot of training time versus GPU count, will be summarized in a report.

###### Model Training Infrastructure & Platform
To meet the requirements of Unit 5, we plan to build a robust model training infrastructure that ensures efficient experiment tracking, job scheduling, and fault tolerance. We will self-host an MLFlow or Weights & Biases (W&B) server on Chameleon to track key training metrics, hyperparameters, and overall model performance. This tracking system will be fully integrated into our training code, allowing us to log and store all experiments for future analysis and improvement.

For scheduling training jobs, we will deploy a Ray cluster to manage job submissions within a continuous training pipeline. To enhance fault tolerance, we aim to incorporate Ray Train, which supports automatic failure recovery and checkpointing to ensure seamless training even in the event of system interruptions. Additionally, we will leverage Ray Tune for hyperparameter tuning, utilizing advanced optimization techniques like Bayesian optimization and ASHA to efficiently explore the search space. A comparative analysis of different tuning methods will also be conducted to determine the most effective approach for our models.

#### Model serving and monitoring platforms

Model serving is a business-critical operation for our Automated Eye Disease Detection System, wherein it must be balanced between high diagnostic accuracy and low-latency inference. With all parts other than the GitHub repository requiring deployment on Chameleon Cloud, serving infrastructure will similarly be completely self-managed on Chameleon Cloud with scalability and efficiency. The model will be containerized with Docker and operated as a microservice for modularity and ease in scalability. Instead of using DagsHub, we will self-host MLflow on a Chameleon instance for experiment tracking, model versioning, and performance monitoring, and model artifacts, logs, and deployment metadata will be kept in Chameleon's persistent storage.

To accelerate inference and optimize use of resources, our system will employ dynamic batching for concurrent image processing, warm-start instances to assist in alleviating cold start latency, and parallel execution to enable dividing inference across different GPUs if necessary. Quantization-aware training and model pruning will be employed to balance performance with high accuracy, leveraging ONNX Runtime or TensorRT to accelerate inference. For scalability, auto-scaling policies will be run using Kubernetes or Docker Swarm, automatically provisioning resources based on workload demand.

To ensure long-term model stability, the system will have self-hosted monitoring using Prometheus and Grafana, tracking key performance metrics such as inference latency (target <200ms), system throughput, and GPU/memory utilization. Furthermore, a custom drift detection module will periodically inspect for new data, identifying potential covariate shift and concept drift that can degrade model performance. Automated retraining pipelines are to be run weekly in Chameleon, using Ray for distributed hyperparameter tuning. New models will be shadow-tested against the current model to measure performance, and canary rollouts will be used for phased deployment to ensure reliability prior to full production release.

By fully self-hosting all components within Chameleon Cloud, this approach ensures compliance with project requirements while delivering a scalable, efficient, and robust AI-powered diagnostic system.

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

The data pipeline for this project is designed to support both offline and online workflows. The Eye Disease Image Dataset, consisting of 16,242 pre-augmented eye images, serves as the primary source of offline data. This dataset is treated as unstructured image data and is stored in a persistent storage provisioned on Chameleon. The persistent storage is mounted to the infrastructure as needed and is used to retain all relevant project artifacts, including datasets, model checkpoints, training logs, and container images that are not tracked in version control.

The offline pipeline is implemented as an ETL (Extract, Transform, Load) process. It ingests the raw dataset, applies preprocessing operations such as image normalization, resizing, and format standardization, and loads the processed data into the storage backend in a format suitable for model training and evaluation. This ensures data consistency across training iterations and enables efficient reuse.

In the online pipeline a streaming data simulation mechanism is implemented. A subset of the dataset is reserved for this purpose to ensure that simulation does not reuse data seen during training or validation and there is no data leakage. These simulation images are sent over time at fixed intervals, mimicking real-time data arrival patterns. The simulation pipeline applies the same preprocessing operations as the offline pipeline, ensuring consistency across inference modes. This real-time data is sent to the deployed model inference service and used to evaluate system behavior under live conditions. The setup enables comprehensive testing of model responsiveness, throughput, and performance monitoring, including the detection of drift and degradation.


#### Continuous X

To ensure the robust and consistent operation of our system, we are implementing a comprehensive continuous integration and continuous deployment (CI/CD) strategy, leveraging the capabilities of GitHub Actions. Firstly, we will employ Infrastructure as Code (IaC) principles, utilizing Terraform to define and manage our cloud infrastructure. This approach allows for version control of our infrastructure configurations, ensuring reproducibility and consistency across environments. Containerization, facilitated by Docker, will be used to encapsulate all application services, promoting portability and simplifying deployment. A GitHub Actions workflow will automate the build, test, and deployment processes, minimizing manual intervention and reducing the potential for human error. We are adopting a microservices architecture to enhance modularity and scalability, enabling independent deployment and management of system components. Furthermore, an immutable infrastructure paradigm will be followed, where changes are implemented through updates to the Terraform configuration rather than direct modifications to deployed resources. All code and configuration artifacts will be managed via Git, providing a complete audit trail and facilitating collaboration.

Automated configuration, using tools such as Ansible or python-chi, will streamline the setup and deployment of software components. We will establish staged deployment environments, specifically staging, canary, and production, to enable rigorous testing and controlled releases. The infrastructure components, including training, serving, and monitoring, will be orchestrated using Terraform. The GitHub Actions workflow will automate the data flow, and monitoring will be integrated to provide proactive feedback. This approach is justified by the need for reproducible infrastructure, consistent deployments, automated workflows, and early detection of issues. It also aligns with the principles and tools discussed in Unit 3, such as Terraform, Docker, and GitHub Actions. We will maintain three distinct environments and aim for weekly retraining and redeployment cycles. All configurations and code will be maintained in a Git repository within GitHub.
