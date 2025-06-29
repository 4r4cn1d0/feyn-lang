#include <stdio.h>

/*******
Name of func: printCombinations
What parameters it takes: int totalCents
What it does: enumerates and prints all combinations of quarters, dimes, nickels, and pennies that sum to totalCents, prioritizing larger denominations first
What it outputs: prints each combination in the format "%d quarter(s), %d dime(s), %d nickel(s), %d pennies\n"
*******/
void printCombinations(int totalCents) {
    for (int quarterCount = totalCents / 25; quarterCount >= 0; quarterCount--) {
        for (int dimeCount = (totalCents - quarterCount * 25) / 10; dimeCount >= 0; dimeCount--) {
            for (int nickelCount = (totalCents - quarterCount * 25 - dimeCount * 10) / 5; nickelCount >= 0; nickelCount--) {
                int pennies = totalCents - quarterCount * 25 - dimeCount * 10 - nickelCount * 5;
                printf("%d quarter(s), %d dime(s), %d nickel(s), %d pennies\n",
                       quarterCount, dimeCount, nickelCount, pennies);
            }
        }
    }
}

/*******
Name of func: main
What parameters it takes: none
What it does: reads an integer amount of cents from stdin and calls printCombinations
What it outputs: returns 0 on success, 1 on input error
*******/
int main(void) {
    int inputCents;
    if (scanf("%d", &inputCents) != 1) {
        return 1;
    }
    printCombinations(inputCents);
    return 0;
}
