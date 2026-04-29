# AI Subscription Saver Agent

An AI Agent for subscription management, spending analysis, price-saving suggestions, and payment risk control.

## Project Overview

This project is a local MVP of an AI-powered personal finance assistant. It helps users analyze bills, identify recurring subscriptions, detect risky payment behavior, and generate money-saving suggestions.

The project focuses on one practical problem: users often subscribe to many digital services, shopping platforms, cloud storage plans, video memberships, and productivity tools, but they may forget renewal dates, miss hidden auto-renewal rules, or pay for duplicated services. This Agent helps users understand their spending more clearly and make safer payment decisions.

## Core Features

- Expense parsing Agent
- Subscription detection Agent
- Price-saving suggestion Agent
- Payment risk control Agent
- Automatic report generation

## Agent Workflow

1. The Expense Parsing Agent reads bill text, order messages, or payment reminders.
2. The Subscription Detection Agent identifies possible recurring payments.
3. The Price-Saving Agent compares current spending with reference low prices.
4. The Risk Control Agent detects suspicious links, abnormal charges, and hidden renewal risks.
5. The Report Agent generates a structured spending and risk analysis report.

## Tech Stack

- Python
- Streamlit
- Pandas

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
