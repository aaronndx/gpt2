import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
import io

def plot_log(logfile_path: str, output_path: str = None):
    """
    Parses a log file and plots the training and validation loss curves.

    The log file is expected to have lines in the format: 'step mode loss'
    e.g., '201 train 6.5397'
    mode can be 'train' for training, and  'val' or 'eval' for validation

    Args:
        logfile_path (str): The path to the log file.
        output_path (str, optional): Path to save the plot image. 
                                     If None, the plot is displayed interactively.
    """
    try:
        # 1. Define the regex pattern for a valid log line
        #    Pattern: (integer) (whitespace) (train|val|eval) (whitespace) (float/int)
        log_pattern = re.compile(r"^\d+\s+(train|val|eval)\s+[\d.]+$")

        # 2. Read the file and keep only the lines that match the pattern
        with open(logfile_path, 'r') as f:
            good_lines = [line for line in f if log_pattern.match(line)]

        # 3. Join the valid lines into a single string and load into pandas
        #    Using io.StringIO is efficient as it treats the string as an in-memory file.
        log_data_string = "".join(good_lines)
        df = pd.read_csv(
            io.StringIO(log_data_string),
            sep =r'\s+',
            header = None,
            names = ['step', 'mode', 'loss']
        )

        # 4. Pivot the DataFrame to separate train and eval losses into columns
        pivot_df = df.pivot(index='step', columns='mode', values='loss')

        # 5. Create the plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 7))

        if 'train' in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df['train'], label='Training Loss', marker='.', linestyle='-')
        
        # Handle both 'val' and 'eval' as possible names
        val_col = 'eval' if 'eval' in pivot_df.columns else 'val'
        if val_col in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[val_col], label='Validation Loss', marker='o', linestyle='--')

        # 6. Customize the plot
        plt.title('Model Loss Over Training Steps', fontsize=16)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.yscale('log') # Log scale is often better for viewing loss
        plt.tight_layout()

        # 7. Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{logfile_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the log file is formatted correctly (e.g., 'step mode loss').")

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Visualize training logs.")
    parser.add_argument("logfile", type=str, help="Path to the log file.")
    parser.add_argument("--output", type=str, help="Path to save the output plot image.", default=None)
    
    args = parser.parse_args()
    plot_log(args.logfile, args.output)
