import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

class TrNSElement:
    """
    Trapezoidal Neutrosophic Set Element class
    Represented as <(a,b,c,d); t,i,f> where:
    - (a,b,c,d) are the trapezoidal values
    - t is the truth membership
    - i is the indeterminacy membership
    - f is the falsity membership
    """
    def __init__(self, a, b, c, d, t, i, f):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.t = t
        self.i = i
        self.f = f

    def __str__(self):
        return f"<({self.a},{self.b},{self.c},{self.d}); {self.t},{self.i},{self.f}>"

    def score(self):
        """Calculate the crisp score value of the trapezoidal neutrosophic element"""
        trapezoid_val = (self.a + self.b + self.c + self.d) / 4
        neutro_val = (2 + self.t - self.i - self.f) / 3
        return trapezoid_val * neutro_val

    def accuracy(self):
        """Calculate the accuracy value"""
        return self.t - self.f

def create_pairwise_comparison_matrix(criteria_names):
    """
    Function to create a pairwise comparison matrix using trapezoidal neutrosophic values
    This is a placeholder - in a real application, you would input actual expert evaluations
    """
    n = len(criteria_names)
    matrix = [[None for _ in range(n)] for _ in range(n)]

    # Diagonal elements are always 1 (criteria compared to itself)
    for i in range(n):
        matrix[i][i] = TrNSElement(1, 1, 1, 1, 1.0, 0.0, 0.0)

    # Example values for demonstration (in a real application, these would come from experts)
    # Security (C1)
    matrix[0][1] = TrNSElement(3, 4, 5, 6, 0.8, 0.2, 0.1)  # Security vs Value
    matrix[0][2] = TrNSElement(1, 2, 3, 4, 0.7, 0.3, 0.2)  # Security vs Intelligence
    matrix[0][3] = TrNSElement(4, 5, 6, 7, 0.9, 0.1, 0.1)  # Security vs Connectivity
    matrix[0][4] = TrNSElement(2, 3, 4, 5, 0.8, 0.2, 0.2)  # Security vs Transparency

    # Value (C2)
    matrix[1][0] = TrNSElement(1/6, 1/5, 1/4, 1/3, 0.8, 0.2, 0.1)  # Value vs Security
    matrix[1][2] = TrNSElement(1/4, 1/3, 1/2, 1, 0.6, 0.3, 0.3)    # Value vs Intelligence
    matrix[1][3] = TrNSElement(1, 2, 3, 4, 0.7, 0.2, 0.2)          # Value vs Connectivity
    matrix[1][4] = TrNSElement(1/5, 1/4, 1/3, 1/2, 0.6, 0.4, 0.3)  # Value vs Transparency

    # Intelligence (C3)
    matrix[2][0] = TrNSElement(1/4, 1/3, 1/2, 1, 0.7, 0.3, 0.2)    # Intelligence vs Security
    matrix[2][1] = TrNSElement(1, 2, 3, 4, 0.6, 0.3, 0.3)          # Intelligence vs Value
    matrix[2][3] = TrNSElement(2, 3, 4, 5, 0.8, 0.1, 0.2)          # Intelligence vs Connectivity
    matrix[2][4] = TrNSElement(1/3, 1/2, 1, 2, 0.7, 0.2, 0.2)      # Intelligence vs Transparency

    # Connectivity (C4)
    matrix[3][0] = TrNSElement(1/7, 1/6, 1/5, 1/4, 0.9, 0.1, 0.1)  # Connectivity vs Security
    matrix[3][1] = TrNSElement(1/4, 1/3, 1/2, 1, 0.7, 0.2, 0.2)    # Connectivity vs Value
    matrix[3][2] = TrNSElement(1/5, 1/4, 1/3, 1/2, 0.8, 0.1, 0.2)  # Connectivity vs Intelligence
    matrix[3][4] = TrNSElement(1/6, 1/5, 1/4, 1/3, 0.7, 0.3, 0.2)  # Connectivity vs Transparency

    # Transparency (C5)
    matrix[4][0] = TrNSElement(1/5, 1/4, 1/3, 1/2, 0.8, 0.2, 0.2)  # Transparency vs Security
    matrix[4][1] = TrNSElement(2, 3, 4, 5, 0.6, 0.4, 0.3)          # Transparency vs Value
    matrix[4][2] = TrNSElement(1/2, 1, 2, 3, 0.7, 0.2, 0.2)        # Transparency vs Intelligence
    matrix[4][3] = TrNSElement(3, 4, 5, 6, 0.7, 0.3, 0.2)          # Transparency vs Connectivity

    return matrix

def calculate_weights_from_pairwise(matrix):
    """Calculate weights from pairwise comparison matrix by converting to crisp values"""
    n = len(matrix)

    # Convert trapezoidal neutrosophic values to crisp scores
    crisp_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            crisp_matrix[i][j] = matrix[i][j].score()

    # Calculate row sums
    row_sums = np.sum(crisp_matrix, axis=1)
    total_sum = np.sum(row_sums)

    # Calculate normalized weights
    weights = row_sums / total_sum

    return weights

def check_consistency(crisp_matrix):
    """Check consistency of the crisp pairwise comparison matrix"""
    n = len(crisp_matrix)

    # Calculate principal eigenvalue and eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(crisp_matrix)
    max_idx = np.argmax(eigenvalues.real)
    max_eigenvalue = eigenvalues[max_idx].real

    # Calculate consistency index (CI)
    CI = (max_eigenvalue - n) / (n - 1)

    # Random Index values for different matrix sizes
    RI_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_values.get(n, 0)

    # Calculate consistency ratio (CR)
    if RI == 0:
        CR = 0
    else:
        CR = CI / RI

    return {'CI': CI, 'CR': CR, 'max_eigenvalue': max_eigenvalue}

def create_decision_matrix(alternatives, criteria):
    """
    Create a decision matrix with trapezoidal neutrosophic values
    This is a placeholder - in a real application, you would input actual evaluations
    """
    n_alternatives = len(alternatives)
    n_criteria = len(criteria)

    decision_matrix = [[None for _ in range(n_criteria)] for _ in range(n_alternatives)]

    # Sample data for Spark
    decision_matrix[0][0] = TrNSElement(7, 8, 9, 9, 0.9, 0.1, 0.1)  # Security
    decision_matrix[0][1] = TrNSElement(6, 7, 8, 9, 0.8, 0.2, 0.2)  # Value
    decision_matrix[0][2] = TrNSElement(8, 9, 9, 10, 0.9, 0.1, 0.0)  # Intelligence
    decision_matrix[0][3] = TrNSElement(5, 6, 7, 8, 0.7, 0.3, 0.2)  # Connectivity
    decision_matrix[0][4] = TrNSElement(6, 7, 8, 9, 0.8, 0.2, 0.1)  # Transparency

    # Sample data for Knime
    decision_matrix[1][0] = TrNSElement(6, 7, 8, 9, 0.8, 0.2, 0.1)  # Security
    decision_matrix[1][1] = TrNSElement(7, 8, 9, 10, 0.9, 0.1, 0.1)  # Value
    decision_matrix[1][2] = TrNSElement(5, 6, 7, 8, 0.7, 0.3, 0.2)  # Intelligence
    decision_matrix[1][3] = TrNSElement(6, 7, 8, 9, 0.8, 0.1, 0.1)  # Connectivity
    decision_matrix[1][4] = TrNSElement(7, 8, 9, 9, 0.8, 0.2, 0.1)  # Transparency

    # Sample data for Hadoop
    decision_matrix[2][0] = TrNSElement(5, 6, 7, 8, 0.7, 0.2, 0.2)  # Security
    decision_matrix[2][1] = TrNSElement(8, 9, 9, 10, 0.9, 0.1, 0.0)  # Value
    decision_matrix[2][2] = TrNSElement(6, 7, 8, 9, 0.8, 0.2, 0.1)  # Intelligence
    decision_matrix[2][3] = TrNSElement(7, 8, 8, 9, 0.8, 0.1, 0.1)  # Connectivity
    decision_matrix[2][4] = TrNSElement(5, 6, 7, 8, 0.7, 0.3, 0.2)  # Transparency

    return decision_matrix

def convert_to_crisp_decision_matrix(matrix):
    """Convert trapezoidal neutrosophic decision matrix to crisp values"""
    rows = len(matrix)
    cols = len(matrix[0])

    crisp_matrix = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            crisp_matrix[i, j] = matrix[i][j].score()

    return crisp_matrix

def perform_topsis(decision_matrix, weights, benefit_criteria):
    """
    Perform TOPSIS method

    Parameters:
    - decision_matrix: numpy array of crisp values
    - weights: numpy array of criteria weights
    - benefit_criteria: list of boolean values (True for benefit, False for cost criteria)

    Returns:
    - Dictionary with TOPSIS results
    """
    # Step 1: Create weighted normalized decision matrix
    rows, cols = decision_matrix.shape

    # Normalize decision matrix
    normalized_matrix = np.zeros((rows, cols))
    for j in range(cols):
        denominator = np.sqrt(np.sum(decision_matrix[:, j] ** 2))
        normalized_matrix[:, j] = decision_matrix[:, j] / denominator

    # Apply weights to normalized matrix
    weighted_normalized_matrix = normalized_matrix * weights

    # Step 2: Determine ideal and negative-ideal solutions
    ideal_solution = np.zeros(cols)
    negative_ideal_solution = np.zeros(cols)

    for j in range(cols):
        if benefit_criteria[j]:  # Benefit criteria (higher is better)
            ideal_solution[j] = np.max(weighted_normalized_matrix[:, j])
            negative_ideal_solution[j] = np.min(weighted_normalized_matrix[:, j])
        else:  # Cost criteria (lower is better)
            ideal_solution[j] = np.min(weighted_normalized_matrix[:, j])
            negative_ideal_solution[j] = np.max(weighted_normalized_matrix[:, j])

    # Step 3: Calculate separation measures
    separation_ideal = np.zeros(rows)
    separation_negative = np.zeros(rows)

    for i in range(rows):
        separation_ideal[i] = np.sqrt(np.sum((weighted_normalized_matrix[i, :] - ideal_solution) ** 2))
        separation_negative[i] = np.sqrt(np.sum((weighted_normalized_matrix[i, :] - negative_ideal_solution) ** 2))

    # Step 4: Calculate relative closeness to the ideal solution
    relative_closeness = separation_negative / (separation_ideal + separation_negative)

    # Step 5: Rank the alternatives
    rank = np.argsort(-relative_closeness)  # Descending order

    return {
        'normalized_matrix': normalized_matrix,
        'weighted_normalized_matrix': weighted_normalized_matrix,
        'ideal_solution': ideal_solution,
        'negative_ideal_solution': negative_ideal_solution,
        'separation_ideal': separation_ideal,
        'separation_negative': separation_negative,
        'relative_closeness': relative_closeness,
        'rank': rank
    }

def print_matrix(matrix, row_labels=None, col_labels=None):
    """Print matrix in a nice format"""
    if isinstance(matrix[0][0], TrNSElement):
        # For TrNSElement matrices, extract scores for visualization
        scores = [[element.score() for element in row] for row in matrix]
        df = pd.DataFrame(scores, index=row_labels, columns=col_labels)
    else:
        # For regular numeric matrices
        df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)

    print(tabulate(df, headers='keys', tablefmt='pretty'))

def main():
    # Define criteria and alternatives
    criteria = ["Security", "Value", "Intelligence", "Connectivity", "Transparency"]
    alternatives = ["Spark", "Knime", "Hadoop"]

    print("\n" + "="*50)
    print("STEP 1: TRAPEZOIDAL NEUTROSOPHIC PAIRWISE COMPARISON")
    print("="*50)

    # Create pairwise comparison matrix for criteria weights
    pairwise_matrix = create_pairwise_comparison_matrix(criteria)

    # Print trapezoidal neutrosophic pairwise matrix
    print("\nPairwise Comparison Matrix (Trapezoidal Neutrosophic, showing crisp scores):")
    print_matrix(pairwise_matrix, criteria, criteria)

    # Convert to crisp scores for weight calculation
    crisp_matrix = np.zeros((len(criteria), len(criteria)))
    for i in range(len(criteria)):
        for j in range(len(criteria)):
            crisp_matrix[i, j] = pairwise_matrix[i][j].score()

    # Check consistency
    consistency = check_consistency(crisp_matrix)
    print(f"\nConsistency Check:")
    print(f"Consistency Index (CI): {consistency['CI']:.4f}")
    print(f"Consistency Ratio (CR): {consistency['CR']:.4f}")
    if consistency['CR'] < 0.1:
        print("The pairwise comparison matrix is consistent (CR < 0.1)\n")
    else:
        print("Warning: The pairwise comparison matrix is inconsistent (CR >= 0.1)\n")

    # Calculate weights from pairwise comparison matrix
    weights = calculate_weights_from_pairwise(pairwise_matrix)
    print("\nCriteria Weights:")
    for i, criterion in enumerate(criteria):
        print(f"{criterion}: {weights[i]:.4f}")

    print("\n" + "="*50)
    print("STEP 2: CREATE DECISION MATRIX")
    print("="*50)

    # Create decision matrix
    decision_matrix = create_decision_matrix(alternatives, criteria)
    print("\nDecision Matrix (Trapezoidal Neutrosophic, showing crisp scores):")
    print_matrix(decision_matrix, alternatives, criteria)

    # Convert decision matrix to crisp values
    crisp_decision_matrix = convert_to_crisp_decision_matrix(decision_matrix)
    print("\nCrisp Decision Matrix:")
    print(pd.DataFrame(crisp_decision_matrix, index=alternatives, columns=criteria).round(4))

    print("\n" + "="*50)
    print("STEP 3: PERFORM TOPSIS")
    print("="*50)

    # All criteria are benefit criteria (higher is better)
    benefit_criteria = [True, True, True, True, True]

    # Perform TOPSIS
    topsis_results = perform_topsis(crisp_decision_matrix, weights, benefit_criteria)

    # Display TOPSIS results
    print("\nNormalized Decision Matrix:")
    print(pd.DataFrame(topsis_results['normalized_matrix'], index=alternatives, columns=criteria).round(4))

    print("\nWeighted Normalized Decision Matrix:")
    print(pd.DataFrame(topsis_results['weighted_normalized_matrix'], index=alternatives, columns=criteria).round(4))

    print("\nIdeal Solution:")
    print(pd.DataFrame([topsis_results['ideal_solution']], columns=criteria).round(4))

    print("\nNegative Ideal Solution:")
    print(pd.DataFrame([topsis_results['negative_ideal_solution']], columns=criteria).round(4))

    print("\nSeparation Measures:")
    separation_df = pd.DataFrame({
        'Distance to Ideal': topsis_results['separation_ideal'],
        'Distance to Negative Ideal': topsis_results['separation_negative']
    }, index=alternatives)
    print(separation_df.round(4))

    print("\nRelative Closeness to Ideal Solution:")
    closeness_df = pd.DataFrame({
        'Relative Closeness': topsis_results['relative_closeness'],
        'Rank': [list(topsis_results['rank']).index(i) + 1 for i in range(len(alternatives))]
    }, index=alternatives)
    print(closeness_df.sort_values('Rank').round(4))

    # Plot the results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(alternatives, topsis_results['relative_closeness'], color=['green', 'blue', 'orange'])
    plt.xlabel('Alternatives')
    plt.ylabel('Relative Closeness to Ideal Solution')
    plt.title('TOPSIS Results for IoT Enterprise Decision Making')

    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

    plt.ylim(0, max(topsis_results['relative_closeness']) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('topsis_results.png')
    print("\nResults visualization saved as 'topsis_results.png'")

    # Final conclusion
    best_alternative = alternatives[topsis_results['rank'][0]]
    best_score = topsis_results['relative_closeness'][topsis_results['rank'][0]]

    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    print(f"\nThe best IoT enterprise solution based on the analysis is: {best_alternative}")
    print(f"Relative closeness score: {best_score:.4f}")
    print("\nRanking of alternatives:")
    for i, idx in enumerate(topsis_results['rank']):
        print(f"{i+1}. {alternatives[idx]} (Score: {topsis_results['relative_closeness'][idx]:.4f})")

if __name__ == "__main__":
    main()