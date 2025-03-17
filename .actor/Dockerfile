# First specify the base Docker image.
FROM apify/actor-python:3.12

# Copy requirements.txt into the Actor image
COPY requirements.txt ./

# Install the packages specified in requirements.txt
RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze

# Copy the remaining files and directories with the source code
COPY . ./

# Use compileall to ensure the runnability of the Actor Python code
RUN python3 -m compileall -q .

# Specify how to launch the source code of your Actor
CMD ["python3", "-m", "src"]
