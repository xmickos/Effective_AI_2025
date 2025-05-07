#include <iostream>
#include <random>
#include <vector>

#include <ttie/ttie.h>

using namespace ttie;

int main()
{
    Tensor a(std::vector<float>({1, 2, 3, 4}));
    Tensor b(std::vector<float>({1, 2, 3, 4}));
    a.reshape({1, 2, 2});
    b.reshape({1, 2, 2});

    Tensor c = a + b;

    std::cout << c << std::endl;
    c.backward();
    std::cout << c << std::endl;
    c.zero_grad();
    std::cout << c << std::endl;
    return 0;
}
