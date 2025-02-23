import os
import subprocess
import sys

def main():
    # Get the path to the shell script
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'AutoDeep.sh')

    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found!")
        sys.exit(1)

    # Run the shell script
    #if "--help" in sys.argv[1] or "-h" in sys.argv[1]:
        #print(f"Core Usage: {os.path.basename(sys.argv[0])} <dirname>")
        #print()
        #print("Further stratifies mirDeep2 outputs into Candidate, Confident, or Falsepositive labels using ML approach")
        #print()
        #print("Input: Directory containing mirDeep2 outputs")
        #print("Output: Directory containing AutoDeep outputs")
        #print()
        #print(f"Model Training Usage: AutoDeep train [Options]")
        #print()
        #print("For further information on training, use:")
        #print(f"AutoDeep train --help")
        #print()
        #print(f"Model Visualization Usage: AutoDeep visualize [Options]")
        #print()
        #print("For further information on visualization, use:")
        #print(f"AutoDeep visualize --help")
        #sys.exit(1)

    if len(sys.argv) <= 1:
        print(f"Usage: {os.path.basename(sys.argv[0])} <dirname>")
        sys.exit(1)
    
    subprocess.run(['bash', script_path] + sys.argv[1:], check=True)

if __name__ == '__main__':
    main()

