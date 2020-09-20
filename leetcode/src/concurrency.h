#pragma once
#include <functional>
class Foo {
    inline const void printFirst() { std::cout << "first"; }
    inline const void printSecond() { std::cout << "second"; }
    inline const void printThird() { std::cout << "third"; }
public:
    Foo() {

    }

    void first(std::function<void()> printFirst) {

        // printFirst() outputs "first". Do not change or remove this line.
        printFirst();
    }

    void second(std::function<void()> printSecond) {

        // printSecond() outputs "second". Do not change or remove this line.
        printSecond();
    }

    void third(std::function<void()> printThird) {

        // printThird() outputs "third". Do not change or remove this line.
        printThird();
    }
};