
## Automated Eye Disease Detection System for Enhanced Clinical Diagnostics

#### Value Proposition:
Our project focuses on developing a machine learning system that seamlessly integrates into the existing diagnostic framework within eye hospitals and clinics. We're not trying to create a new business; instead, we're aiming to augment the capabilities of current medical practices. This system will analyze retinal images, providing ophthalmologists with an automated tool to aid in the early and accurate detection of a range of eye diseases, including Retinitis Pigmentosa, Retinal Detachment, Pterygium, Myopia, Macular Scar, Glaucoma, Disc Edema, Diabetic Retinopathy, and Central Serous Chorioretinopathy. 

#### Current Status Quo:
Currently, the diagnosis of these diseases relies heavily on manual examination by specialists, a process that can be time-consuming and subjective. Even with advanced imaging technologies like fundus photography and OCT, the interpretation of these images demands significant expertise, often leading to delays in diagnosis. Our system aims to streamline this process, providing quick and consistent analysis to support clinicians in making informed decisions.

#### Business Metric:
The success of our system will be judged on its ability to improve the efficiency and accuracy of eye disease diagnosis. To quantify this, we'll focus on several key business metrics. First, we'll measure the reduction in "time to diagnosis," comparing the time taken to reach a diagnosis with our automated system versus the current manual process. Secondly, we'll assess the "diagnostic accuracy" by determining the concordance rate between our system's predictions and the gold standard diagnosis provided by ophthalmologists. We'll also examine the system's "sensitivity and specificity," ensuring it accurately identifies both positive and negative cases. Additionally, we'll track the "throughput" of our system, which refers to the number of images processed per unit time, to gauge its efficiency in a clinical setting. Finally, and perhaps most importantly, we will focus on the "reduction of false negatives". A false negative could have dire consequences for a patient, and reducing these instances is a paramount concern for our system’s success. By focusing on these metrics, we aim to demonstrate the tangible benefits of our machine learning system in improving patient care and clinical workflows.

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for                 | Link to their commits in this repo |
|---------------------------------|---------------------------------|------------------------------------|
| Rushabh Bhatt                   | CI/CD, Infrastructure as Code,Version Control, Proactive, Monitoring and Logging  | https://github.com/rushxbh910/Automated-Eye-Disease-Detection-System/commit/af56db8
|
| Shruti Bora                     |                                 |                                    |
| Aryan Ajmera                    |                                 |                                    |
| Vaibhav Rouduri                 |                                 |                                    |



### System diagram

![System Diagram](System-Diagram.png)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

To ensure the robust and consistent operation of our system, we are implementing a comprehensive continuous integration and continuous deployment (CI/CD) strategy, leveraging the capabilities of GitHub Actions. Firstly, we will employ Infrastructure as Code (IaC) principles, utilizing Terraform to define and manage our cloud infrastructure. This approach allows for version control of our infrastructure configurations, ensuring reproducibility and consistency across environments. Containerization, facilitated by Docker, will be used to encapsulate all application services, promoting portability and simplifying deployment. A GitHub Actions workflow will automate the build, test, and deployment processes, minimizing manual intervention and reducing the potential for human error. We are adopting a microservices architecture to enhance modularity and scalability, enabling independent deployment and management of system components. Furthermore, an immutable infrastructure paradigm will be followed, where changes are implemented through updates to the Terraform configuration rather than direct modifications to deployed resources. All code and configuration artifacts will be managed via Git, providing a complete audit trail and facilitating collaboration.

Automated configuration, using tools such as Ansible or python-chi, will streamline the setup and deployment of software components. We will establish staged deployment environments, specifically staging, canary, and production, to enable rigorous testing and controlled releases. The infrastructure components, including training, serving, and monitoring, will be orchestrated using Terraform. The GitHub Actions workflow will automate the data flow, and monitoring will be integrated to provide proactive feedback. This approach is justified by the need for reproducible infrastructure, consistent deployments, automated workflows, and early detection of issues. It also aligns with the principles and tools discussed in Unit 3, such as Terraform, Docker, and GitHub Actions. We will maintain three distinct environments and aim for weekly retraining and redeployment cycles. All configurations and code will be maintained in a Git repository within GitHub.