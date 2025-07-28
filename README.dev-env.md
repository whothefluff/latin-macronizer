# Latin Macronizer (Containerized Version)

This fork containerizes the original [Latin Macronizer by Alatius](https://github.com/Alatius/latin-macronizer) using Docker. The goal is to provide a stable, repeatable, and easy-to-use environment for building, training, and running the macronizer application.

## Improvements

This project is structured around a three-step philosophy over the original's installation and usage requirements to make sure it can be preserved easily:

### 1. A Pre-Built "Golden Base" Environment
The primary `Dockerfile` in this repository creates a **"Golden Base"** image. This image contains all the slow-to-compile system dependencies (`Morpheus`, `RFTagger`) and the Python environment. It has completed the entire installation process **except for the final training step**. The compilation work is done once and captured in this stable, reusable image.

### 2. Easily Create Portable Application Images
From the "Golden Base" image, you can easily create any number of self-contained, **Final Application Images**.
*   **No Extra System Dependencies:** The only thing you need to provide is the **treebank data**.
*   **The Result is a Portable Application:** The output of the training process is a final, locked-down image containing all trained models and the database. **This image is a single artifact that can be easily saved, shared, and run on any machine with Docker.** You can create different versions of the application (e.g., one trained on classical texts, another on medieval texts) simply by using different treebank data. *Note: The `train-rftagger.sh` script is tailored for the data format found in the treebank_data/v1.6 repository.*

### 3. A Clear Path for Debugging Final Images
It provides a straightforward workflow to **debug the Python code** running inside a final, trained application image. By attaching a code editor like VS Code to the running container, you can set breakpoints, inspect variables, and step through the code live.

All original installation instructions in `INSTALL.txt` can be disregarded in favor of the workflow described below.

## The Workflow in a Nutshell

The core idea is to separate the slow, complex build process from the fast, iterative training process.

1.  **Golden Base Image (`whothefluff/latin-macronizer:dev-env-v1`)**: A stable, locked image that contains all the compiled tools (Morpheus, RFTagger) but has **not** been trained. You build this once and reuse it many times.
2.  **Final Application Image (e.g., `whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1`)**: A final, runnable image created from the Golden Base. This image **includes** the trained models and database, making it a self-contained, distributable application.

### Step 1: Prerequisites

Ensure you have [Docker](https://www.docker.com/get-started) installed and running on your system.

### Step 2: Build the Golden Base Image

This step compiles all the tools. You only need to do this once, or whenever you change the underlying `Dockerfile`.

1.  Clone this repository to your local machine.
2.  Navigate into the repository's root directory in your terminal.
3.  Run the build command:

    ```bash
    # This command builds the image and tags it with your desired name
    docker build -t whothefluff/latin-macronizer:dev-env-v1 .
    ```

You now have a stable `whothefluff/latin-macronizer:dev-env-v1` image ready for the training process.

### Step 3: Train and Create a Final Application Image

This is the interactive workflow you will use every time you want to train the model with new or updated treebank data.

1.  **Start a temporary container** from your Golden Base image. We give it a name so we can easily save its state later.

    ```bash
    docker run -it --name latin-training-session whothefluff/latin-macronizer:dev-env-v1
    ```

    You are now inside the container's shell as the `appuser`, at the `/app` prompt.

2.  **Perform the final training steps** inside the container:

    ```bash
    # You are now inside the container

    # 1. Download the treebank data (or copy your own custom data)
    git clone https://github.com/Alatius/treebank_data.git

    # 2. Convert the corpus and train RFTagger
    ./train-rftagger.sh

    # 3. Initialize the macronizer's database
    python macronize.py --initialize

    # 4. (Optional but recommended) Run the test to confirm it works
    python macronize.py --test
    # You should see the success message: "Ō orbis terrārum tē salūtō!"
    ```

3.  **Exit the container** by typing `exit` or pressing `Ctrl+D`.

4.  **Save the trained state** by committing the container. This creates your new, self-contained application image. For this example, we'll tag it as `dev-env-v1-alatius-treebank-v1`.

    ```bash
    # Syntax: docker commit <container_name> <new_image_name:tag>
    docker commit latin-training-session whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1
    ```

5.  **Clean up** by removing the temporary container, which is no longer needed.

    ```bash
    docker rm latin-training-session
    ```

You have now created a final, distributable image named `whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1` containing the fully configured and trained application.

## How to Use Your Trained Image

You can now use your final image to run the macronizer from any terminal, without needing to enter the container interactively.

*   To run the built-in test:

    ```bash
    docker run --rm whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1 python macronize.py --test
    ```

*   To macronize a string:

    ```bash
    echo "puer in via ambulat" | docker run -i --rm whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1 python macronize.py
    ```
    Output:
    ```
    Puer in viā ambulat.
    ```

## Naming and Versioning Your Images

It is highly recommended to use descriptive tags for your final images. If you train the model with a different set of treebank data, for example, you could name the resulting image accordingly:

```bash
docker commit latin-training-session whothefluff/latin-macronizer:dev-env-v1-trained-with-propertius-v1
```

This allows you to maintain multiple, distinct, trained versions of the application simultaneously.

## Debugging with VS Code

Debugging this application requires a hybrid approach. We use a volume mount to link your local source code, but first, we must copy all the essential **trained files and generated modules** from your final image into your local project directory.

This process has a **one-time setup** per trained image, after which you can debug repeatably without further errors.

### One-Time Setup: Prepare Your Local Directory

The training and initialization scripts create four critical files that must be present for debugging. Run the following commands once to copy all of them from your trained image into your local project folder.

1.  **Copy the Database:**
    ```bash
    docker cp $(docker create whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1):/app/macronizer.db .
    ```

2.  **Copy the RFTagger Model:**
    ```bash
    docker cp $(docker create whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1):/app/rftagger-ldt.model .
    ```

3.  **Copy the Generated Lemmas Module:**
    ```bash
    docker cp $(docker create whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1):/app/lemmas.py .
    ```

4.  **Copy the Generated Endings Module:**
    ```bash
    docker cp $(docker create whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1):/app/macronized_endings.py .
    ```

5.  **(Recommended) Update `.dockerignore`:**
    To prevent these generated files from being included in future image builds, add their names to your `.dockerignore` file.

    **.dockerignore**
    ```
    # Ignore generated files that should not be part of the build context
    macronizer.db
    rftagger-ldt.model
    lemmas.py
    macronized_endings.py
    
    # Ignore local dev files
    .vscode/
    ```

Your local folder now contains a complete set of the artifacts needed for debugging.

### Debugging Your Script

You can now use `docker run` with a volume mount. Because your local folder contains all the required models, databases, and modules, the script will find everything it needs.

1.  **Launch the Container in Debug Mode:**
    Use the appropriate command for your use case.

    **To debug with piped input:**
    ```bash
    echo "puer in via ambulat" | docker run --rm -i \
      -p 5678:5678 \
      -v "$(pwd)":/app \
      whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1 \
      python -m debugpy --wait-for-client --listen 0.0.0.0:5678 macronize.py
    ```

    **To debug with arguments (e.g., --test):**
    ```bash
    docker run --rm -it \
      -p 5678:5678 \
      -v "$(pwd)":/app \
      whothefluff/latin-macronizer:dev-env-v1-alatius-treebank-v1 \
      python -m debugpy --wait-for-client --listen 0.0.0.0:5678 macronize.py --test
    ```

2.  **Attach VS Code:**
    Once the command is running and waiting, go to the **Run and Debug** view in VS Code, select **"Python: Attach to Docker Container"**, and click the green play button. Execution will now stop at your breakpoints.