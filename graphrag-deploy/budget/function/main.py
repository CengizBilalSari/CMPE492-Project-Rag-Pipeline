import base64
import json
import os

from googleapiclient import discovery
import functions_framework


PROJECT_ID = os.getenv("GCP_PROJECT")
PROJECT_NAME = f"projects/{PROJECT_ID}"


@functions_framework.cloud_event
def stop_billing(cloud_event):
    """Cloud Function triggered by Pub/Sub when budget threshold is reached."""
    pubsub_data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    pubsub_json = json.loads(pubsub_data)
    cost_amount = pubsub_json["costAmount"]
    budget_amount = pubsub_json["budgetAmount"]

    print(f"Budget notification: cost={cost_amount}, budget={budget_amount}")

    if cost_amount <= budget_amount:
        print(f"No action needed. (Cost {cost_amount} <= Budget {budget_amount})")
        return

    if PROJECT_ID is None:
        print("ERROR: GCP_PROJECT environment variable not set")
        return

    billing = discovery.build("cloudbilling", "v1", cache_discovery=False)
    projects = billing.projects()

    if _is_billing_enabled(PROJECT_NAME, projects):
        _disable_billing_for_project(PROJECT_NAME, projects)
    else:
        print("Billing already disabled")


def _is_billing_enabled(project_name, projects):
    """Check whether billing is enabled for the project."""
    try:
        res = projects.getBillingInfo(name=project_name).execute()
        return res.get("billingEnabled", False)
    except Exception as e:
        print(f"Unable to determine billing status: {e}")
        return True  # Assume enabled to be safe


def _disable_billing_for_project(project_name, projects):
    """Disable billing by removing the billing account."""
    body = {"billingAccountName": ""}
    try:
        res = projects.updateBillingInfo(name=project_name, body=body).execute()
        print(f"Billing DISABLED: {json.dumps(res)}")
    except Exception as e:
        print(f"Failed to disable billing: {e}")
        print("Check that the Cloud Function service account has billing.admin role")
