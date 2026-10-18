import logging
from datetime import date

from langchain.tools import tool

from agent.data import users_data
from agent.objects import PurchaseItem


@tool
def get_user_purchase_history(user: str) -> dict:
    """
    This tool takes the username and return dictionary of the user
    purchase history. If the user exists.
    Args:
        user (str): username

    Returns:
        dict: purchase history
    """

    logging.info(f"Using get_user_purchase_history tool on user: {user}")

    if users_data.get(user):
        return users_data[user]
    else:
        return {"status": "There was no username found"}


@tool
def calculate_right_refund(user: str, items: list[PurchaseItem]) -> dict:
    """
    This tool takes a list of the items to be returned and username and
    will return how much refund is worth
    Args:
        user (str): username
        items (list[PurchaseItem]): list of purchase items

    Returns:
        dict: The status of refund if username exist else display default message if username not found
    """

    logging.info(f"Using calculate_right_refund tool to get user's: {user} refund")

    if users_data.get(user):
        total_refund = 0

        for item in items:
            if users_data[user].get(item.name):
                total_refund += users_data[user][item.name]["price"]

        return {"status": f"The value of the refund is: {total_refund}"}
    else:
        return {"status": "There was no username found"}


@tool
def issue_refund_ticket(refund_amount: int, user: str) -> str:
    """
    This tool will send the ticket to the right team to
    return the refund
    Args:
        refund_amount (int): refund amount
        user (str): user name

    Returns:
        str: issued ticket with the refund amount
    """

    logging.info(f"Using issue_refund_ticket for user: {user}")

    return f"""
    A refund ticket has been issued successfully with the amount of: {refund_amount}
    """


@tool
def get_today_date(user: str) -> str:
    """This will return today's date

    Args:
        user (str): username

    Returns:
        str: today's date
    """

    logging.info(f"Using get_today_date tool for user: {user}")

    return date.today()
