#include <stdio.h>

/*******
Name of func: printCombinations
What parameters it takes: int totalCents
What it does: enumerates and prints all combinations of quarters, dimes, nickels, and pennies that sum to totalCents, prioritizing larger denominations first
What it outputs: prints each combination in the format "%d quarter(s), %d dime(s), %d nickel(s), %d pennies\n"
*******/
void printCombinations(int totalCents) {
    int maxQuarters = totalCents / 25;
    for (int quarterCount = maxQuarters; quarterCount >= 0; quarterCount--) {
        int centsAfterQuarters = totalCents - quarterCount * 25;
        int maxDimes = centsAfterQuarters / 10;
        for (int dimeCount = maxDimes; dimeCount >= 0; dimeCount--) {
            int centsAfterDimes = centsAfterQuarters - dimeCount * 10;
            int maxNickels = centsAfterDimes / 5;
            for (int nickelCount = maxNickels; nickelCount >= 0; nickelCount--) {
                int pennies = centsAfterDimes - nickelCount * 5;
                printf("%d quarter(s), %d dime(s), %d nickel(s), %d pennie(s)\n",
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
