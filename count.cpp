#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <cctype>

int main() {
    std::ifstream file("corpus.txt", std::ios::binary);

    if (!file) {
        std::cerr << "Dosya acilamadi!\n";
        return 1;
    }

    constexpr std::size_t BUFFER_SIZE = 16 * 1024 * 1024;
    std::vector<char> buffer(BUFFER_SIZE);

    std::uint64_t word_count = 0;
    std::uint64_t line_count = 0;

    bool in_word = false;

    while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
        std::streamsize count = file.gcount();

        for (std::streamsize i = 0; i < count; ++i) {
            unsigned char c = static_cast<unsigned char>(buffer[i]);

            // Satır say
            if (c == '\n') {
                ++line_count;
            }

            // Kelime say
            if (std::isspace(c)) {
                in_word = false;
            }
            else if (!in_word) {
                ++word_count;
                in_word = true;
            }
        }
    }

    std::cout << "Kelime sayisi : " << word_count << '\n';
    std::cout << "Satir sayisi  : " << line_count << '\n';
}
