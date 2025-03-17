# AI Job Finder

> **⚠️ IMPORTANT**: This Actor requires the LinkedIn Jobs Scraper to function properly. You must rent it from [https://apify.com/bebity/linkedin-jobs-scraper](https://apify.com/bebity/linkedin-jobs-scraper) before using this Actor.

An Apify Actor that uses AI to analyze your CV and/or job search prompt to find relevant jobs on LinkedIn.

## Overview

This Actor combines the power of Large Language Models (LLMs) with LinkedIn job scraping to provide highly relevant job matches. It works by:

1. Reading your CV and/or job search prompt
2. Using AI (OpenAI, Claude, or Gemini) to extract key job search parameters
3. Searching LinkedIn for matching jobs using these parameters
4. Returning detailed job listings that match your profile and preferences

## Features

- **CV Analysis**: Upload your CV/resume and the Actor will analyze it to understand your experience, skills, and career level
- **Natural Language Search**: Describe the job you're looking for in plain language
- **Multiple AI Providers**: Choose between OpenAI, Claude, or Gemini for analysis
- **Detailed Job Listings**: Get complete job details including description, requirements, company information, and application links

## Requirements

- An Apify account
- The LinkedIn Jobs Scraper Actor rented from [apify.com/bebity/linkedin-jobs-scraper](https://apify.com/bebity/linkedin-jobs-scraper)
- An Apify API key
- At least one API key for an LLM provider (OpenAI, Claude, or Gemini)

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cv` | File | Your CV/resume in PDF, DOCX, or TXT format (optional if prompt is provided) |
| `prompt` | Text | Natural language description of the job you're looking for (optional if CV is provided) |
| `llm_settings` | Object | Configure which LLM provider and model to use |
| `api_keys` | Object | API keys for the LLM providers |
| `apify_api_key` | String | Your Apify API key for calling the LinkedIn Jobs Scraper |
| `linkedin_search_params` | Object | Manual override for LinkedIn search parameters (optional) |
| `proxy` | Object | Proxy configuration for LinkedIn scraping |

### LLM Settings

```json
{
  "provider": "openai", // One of: "openai", "claude", "gemini"
  "model": "gpt-4o"     // Model to use, depends on the provider
}
```

### API Keys

```json
{
  "openai": "sk-...",  // OpenAI API key
  "claude": "sk-...",  // Claude API key (if using Claude)
  "gemini": "..."      // Gemini API key (if using Gemini)
}
```

## Example Usage

### With CV Only

Upload your CV and the Actor will analyze it to find matching jobs based on your experience and skills.

### With Prompt Only

```
I'm looking for remote Senior Python Developer roles at tech startups, preferably in the AI field. I have 5 years of experience and am looking for full-time positions posted in the last week.
```

### With Both CV and Prompt

Combining both inputs provides the most accurate results. The Actor will extract information from your CV and use the prompt to refine the search criteria.

## Output

The Actor returns an array of job listings with detailed information about each job, including:

- Job title and company
- Location and work arrangement (remote, on-site, hybrid)
- Full job description
- Application link
- Posting date
- Company details
- Salary information (when available)

## How It Works

1. **Input Validation**: Ensures at least one of CV or prompt is provided
2. **CV Processing**: If a CV is provided, it's analyzed by the selected LLM to extract relevant information
3. **Prompt Processing**: If a prompt is provided, it's analyzed to extract search parameters
4. **Parameter Consolidation**: Parameters from both sources are combined (with prompt taking precedence)
5. **LinkedIn Scraping**: The extracted parameters are used to search for jobs on LinkedIn
6. **Results**: Matching job listings are returned

## Supported LinkedIn Search Parameters

| Parameter | Description | Format |
|-----------|-------------|--------|
| `title` | Job title to search for | String |
| `location` | Geographic location (defaults to "United States" if not specified) | String |
| `experienceLevel` | Experience level | "1" (Internship) to "5" (Director) |
| `workType` | Work arrangement | "1" (On-Site), "2" (Remote), "3" (Hybrid) |
| `contractType` | Employment type | "F" (Full-Time), "P" (Part-Time), etc. |
| `publishedAt` | Time frame | "r86400" (24h), "r604800" (week), "r2592000" (month) |
| `companyName` | List of companies | Array of strings |
| `companyId` | List of LinkedIn company IDs | Array of strings |
| `rows` | Number of results to return | Integer |
