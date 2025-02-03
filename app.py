#!/usr/bin/env python3
import aws_cdk as cdk
from research_helper_stack import ResearchHelperStack

app = cdk.App()

ResearchHelperStack(app, "ResearchHelperStack", 
    env=cdk.Environment(
        account="034129651538", 
        region="eu-west-1"
    )
)

app.synth()
