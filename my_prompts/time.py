def read_numbers_from_file(filename):
    with open(filename, "r") as file:
        numbers = [float(line.strip()) for line in file]
    return numbers


def calculate_statistics(numbers):
    if not numbers:
        return None, None, None

    total = sum(numbers)
    count = len(numbers)
    average = total / count
    minimum = min(numbers)
    maximum = max(numbers)

    return average, minimum, maximum


def main():
    filename = "k_matmul_time.txt"
    numbers = read_numbers_from_file(filename)

    average, minimum, maximum = calculate_statistics(numbers)

    print(f"avg: {average:.5f}")
    print(f"min: {minimum:.5f}")
    print(f"max: {maximum:.5f}")


if __name__ == "__main__":
    main()
