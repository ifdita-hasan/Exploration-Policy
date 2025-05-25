import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict

def main():
    # Directory containing all PPO experiment results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Dictionary to store screenshots for each entropy coefficient
    screenshots = defaultdict(list)
    
    # Process each PPO directory
    for dir_name in os.listdir(data_dir):
        if dir_name.startswith('ppo-'):
            # Get entropy coefficient from directory name
            entropy_coeff = float(dir_name.split('-')[1])
            
            # Get all PNG files in the directory
            dir_path = os.path.join(data_dir, dir_name)
            png_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')])
            
            # Only process directories with exactly 5 PNG files
            if len(png_files) == 5:
                # Skip the first file (plot), keep the rest
                screenshots[entropy_coeff] = [os.path.join(dir_path, f) for f in png_files[1:]]
    
    # Sort entropy coefficients
    entropy_coeffs = sorted(screenshots.keys())
    
    # Split entropy coefficients into two parts
    entropy_coeffs_first = entropy_coeffs[:5]
    entropy_coeffs_second = entropy_coeffs[5:]
    
    # Function to create a plot for a subset of coefficients
    def create_plot(coeffs, save_name):
        # Create figure with appropriate number of rows
        num_rows = len(coeffs)
        # Adjust height based on number of rows (each row needs about 1.4 inches)
        fig_height = max(1.4 * num_rows, 3.5)  # Reduced minimum height
        fig, axes = plt.subplots(num_rows, 3, figsize=(7, fig_height))  # Reduced width
        
        # Calculate the x position for labels (absolute position relative to figure)
        # We want the labels to be positioned at x=0.025 of the figure width
        fig_x = 0.035 # 2.5% of figure width
        
        # Add entropy coefficient labels on the left
        for i, entropy_coeff in enumerate(coeffs):
            # Format as c₂ = value
            label = f'$c_2 = {entropy_coeff:.4f}$'
            # Calculate y position relative to figure (centered vertically in each row)
            # Start from top instead of bottom
            fig_y = 1 - (i + 0.5) / num_rows  # +0.5 to center vertically
            fig.text(fig_x, fig_y, label,
                     fontsize=8,  # Reduced font size
                     ha='right', va='center',
                     rotation='vertical')
        
        # Remove unused axes
        # Only remove if we have fewer rows than we created
        if num_rows < len(axes):
            for i in range(num_rows, len(axes)):
                for j in range(3):
                    axes[i, j].axis('off')
        
        # Remove all spacing
        plt.subplots_adjust(hspace=0, wspace=0)
        
        # Plot each screenshot
        for i, entropy_coeff in enumerate(coeffs):
            screenshot_paths = screenshots[entropy_coeff]
            for j in range(3):
                img = mpimg.imread(screenshot_paths[j+1])
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
        
        # Remove padding
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0.025,  # Reduced left padding
                           bottom=0.005,  # Reduced bottom padding
                           top=0.995,    # Reduced top padding
                           right=0.995)  # Reduced right padding
        
        # Save and show the plot
        save_path = os.path.join(data_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600)  # No padding
        plt.show()
        print(f"Plot saved to: {save_path}")
    
    # Create first plot (first 5 coefficients)
    if entropy_coeffs_first:
        create_plot(entropy_coeffs_first, 'trajectory_comparison_first.png')
    
    # Create second plot (remaining coefficients)
    if entropy_coeffs_second:
        create_plot(entropy_coeffs_second, 'trajectory_comparison_second.png')

if __name__ == "__main__":
    main()