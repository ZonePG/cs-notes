# Lab Checkpoint 0: networking warmup

##  Writing a network program using an OS stream socket

我们可以通过 OS 提供的功能，创建可靠双向字节流(**reliable bidirectional byte**, 又称为流套接字 **stream socket**)。在这个热身实验中，我们通过可靠双向字节流获取网页数据。

流套接字看起来像是普通的文件描述符，当两个流套接字连接的时候，某一方流套接字的输入最终会在另一方的流套接字中输出。

事实上，网络传输过程中并不保证能够传输可靠的字节流，因此在传输过程中，因此数据在网络传输过程中，可能会发生如下四种问题：
- lost，丢失
- delivered out of order，乱序
- delivered with the contents altered，更改
- duplicated and delivered more than once，重复

因此，连接两端之间的主机的 OS 通常会提供 TCP 协议，从而保证传输可靠的字节流。而在后续的 lab 中，我们将动手实现自己的 TCP 协议。

### Writing webget

这一部分我们将使用 OS 提供的 TCP 接口支持，获取网页数据，我们只需要实现 `get_URL()` 方法即可。

阅读 **TCPSocket** 提供的方法：
- `connect()` 方法向一个 host 发起连接请求，
- `write()` 方法向 host 写入发送请求。
- `read()` 方法获取 host 返回的数据。

需要注意的是：
- 客户端写入请求时，每一行以 "\r\n" 结尾。
- 客户端写入请求时，"Connection: close"，通知服务器端不再等待客户端发送请求，并将请求的数据返回给客户端。
- 确保输出服务器传送过来的所有数据，仅调用一次 `read()` 方法是不够的。

最终实现如下：
```c++
void get_URL(const string &host, const string &path) {
    // Your code here.

    // You will need to connect to the "http" service on
    // the computer whose name is in the "host" string,
    // then request the URL path given in the "path" string.

    // Then you'll need to print out everything the server sends back,
    // (not just one call to read() -- everything) until you reach
    // the "eof" (end of file).

    // cerr << "Function called: get_URL(" << host << ", " << path << ").\n";
    // cerr << "Warning: get_URL() has not been implemented yet.\n";

    TCPSocket tcp_socket;
    tcp_socket.connect(Address(host, "http"));
    tcp_socket.write("GET " + path + " HTTP/1.1\r\n");
    tcp_socket.write("HOST: " + host + "\r\n");
    tcp_socket.write("Connection: close\r\n\r\n");

    while (!tcp_socket.eof()) {
      cout << tcp_socket.read();
    }

    tcp_socket.close();
}
```

## An in-memory reliable byte stream

这一部分是实现**输入端写入数据，输出端读取数据**的字节流缓冲区。缓冲区大小是有限的，当缓冲区容量满时，不能再写入数据。当输入端不再写入数据时，并且输出端已经将缓冲区的数据读取完毕，也就是缓冲区为空，此时缓冲区的状态就是 **EOF**(end of file)。

虽然字节流缓冲区是有限的，但是只要满足一定的条件，它可以处理无限长的数据，比如：缓冲区大小为 1，如果每次只输入 1 字节的数据，并且输出端在输入端写入下一个字节之前读取这一字节数据，那么缓冲区就可以处理无限长的数据，只要输入端仍在写入数据。

### 具体实现

虽然缓冲区是先进先出结构，由于缓冲区要有`peek_output(const size_t len)`方法，也就是查看输出端字节长为 len 的数据，因此使用 **std::deque** 作为缓冲区的数据结构更为合适。

声明如下：

```c++
// byte_stream.hh
class ByteStream {
  private:
    // Your code here -- add private members as necessary.
    size_t _capacity{};
    size_t _remaining_capacity{};
    std::deque<char> _buffer{};
    bool _input_ended{};
    size_t _bytes_written{};
    size_t _bytes_read{};
};
```

实现如下：
```c++
size_t ByteStream::write(const string &data) {
    size_t write_len = min(data.size(), _remaining_capacity);
    _buffer.insert(_buffer.end(), data.begin(), data.begin() + write_len);
    _bytes_written += write_len;
    _remaining_capacity -= write_len;
    return write_len;
}

//! \param[in] len bytes will be copied from the output side of the buffer
string ByteStream::peek_output(const size_t len) const {
    size_t peek_len = min(buffer_size(), len);
    return string(_buffer.begin(), _buffer.begin() + peek_len);
}

//! \param[in] len bytes will be removed from the output side of the buffer
void ByteStream::pop_output(const size_t len) {
    size_t pop_len = min(buffer_size(), len);
    _buffer.erase(_buffer.begin(), _buffer.begin() + pop_len);
    _bytes_read += pop_len;
    _remaining_capacity += pop_len;
}

//! Read (i.e., copy and then pop) the next "len" bytes of the stream
//! \param[in] len bytes will be popped and returned
//! \returns a string
std::string ByteStream::read(const size_t len) {
    string read_str = peek_output(len);
    pop_output(len);
    return read_str;
}

void ByteStream::end_input() { _input_ended = true; }

bool ByteStream::input_ended() const { return _input_ended; }

size_t ByteStream::buffer_size() const { return _buffer.size(); }

bool ByteStream::buffer_empty() const { return _buffer.size() == 0; }

bool ByteStream::eof() const { return buffer_empty() && input_ended(); }

size_t ByteStream::bytes_written() const { return _bytes_written; }

size_t ByteStream::bytes_read() const { return _bytes_read; }

size_t ByteStream::remaining_capacity() const { return _remaining_capacity; }
```