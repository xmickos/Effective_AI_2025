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

    Tensor c = a + (b + a);

    std::cout << "c:" << std::endl;
    c.backward();
    std::cout << "backwarded" << std::endl;
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::for_each(b.get_grad().begin(), b.get_grad().end(), [](float t){ std::cout << t << " "; });
    return 0;
}
