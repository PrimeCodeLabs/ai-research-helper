from aws_cdk import (
    Stack,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_ec2 as ec2,
    aws_ecs_patterns as ecs_patterns,
    CfnOutput,
)
from constructs import Construct

class ResearchHelperStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. ECS Cluster (no VPC)
        cluster = ecs.Cluster(self, "ResearchHelperCluster")

        # 2. Retrieve the SSM Parameter (Hugging Face token)
        hf_token_parameter = ssm.StringParameter.from_string_parameter_name(
            self,
            "HfHubTokenParam",
            string_parameter_name="/research-helper/hf-hub-token"
        )

        # 3. Create the task definition
        task_definition = ecs.FargateTaskDefinition(
            self, "ResearchHelperTask",
            execution_role=iam.Role(
                self,
                "EcsExecutionRole",
                assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy"),
                    iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMReadOnlyAccess")
                ]
            ),
            task_role=iam.Role(
                self,
                "EcsTaskRole",
                assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
            ),
            cpu=512,  # Set CPU here in the task definition
            memory_limit_mib=1024  # Set memory here in the task definition
        )

        task_definition.add_container(
                "ResearchHelperContainer",
                image=ecs.ContainerImage.from_registry(
                    "034129651538.dkr.ecr.eu-west-1.amazonaws.com/research-helper-repo:latest"
                ),
                environment={
                    "HUGGINGFACEHUB_API_TOKEN": hf_token_parameter.string_value  # Using string_value instead of value
                },
                port_mappings=[ecs.PortMapping(container_port=8000)]
            )

        # 4. Create Application Load Balancer
        load_balancer = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "ResearchHelperAppLoadBalancer",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            task_definition=task_definition,
            public_load_balancer=True,  # Set to True to allow public access
            assign_public_ip=True,
            load_balancer_name="ai-research-helper"
        )

        # 5. Output ECS service details
        CfnOutput(
            self,
            "ECSServiceDetails",
            value=f"Service ARN: {load_balancer.service.service_arn}"
        )

        # 6. Output Load Balancer DNS
        CfnOutput(
            self,
            "LoadBalancerDNS",
            value=f"Load Balancer DNS: {load_balancer.load_balancer.load_balancer_dns_name}"
        )
