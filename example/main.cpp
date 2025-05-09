#include <iostream>
#include <random>
#include <vector>

#include <ttie/ttie.h>

using namespace ttie;

int main()
{
    Tensor a(std::vector<float>({4, 3, 2, 1}), true);
    Tensor b(std::vector<float>({1, 2, 3, 4}), true);
    a.reshape({1, 2, 2});
    b.reshape({1, 2, 2});

    Tensor c = a / b;

    std::cout << "c:" << std::endl;
    c.backward();
    std::cout << "backwarded" << std::endl;
    std::cout << a[{1, 1, 1}];
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    return 0;
}
