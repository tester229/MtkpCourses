#include <iostream>

int main()
{
	int mass [50];
	for (int i = 0; i < 50; i++) {
		mass[i] = rand() % 50;
	}

	for (int i = 0; i < 50; i++) {
		for (int j = 1; j < 50; j++) {
			if (mass[j - 1] < mass[j])
				std::swap(mass[j], mass[j - 1]);
}
	}

}
