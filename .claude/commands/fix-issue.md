Find and fix issue #$ARGUMENTS. Follow these steps:

1. Checkout a feature branch to do the issue in 
2. Understand the issue described in the ticket 
3. Locate the relevant code in our codebase 
4. Implement a solution that addresses the root cause 
5. Add appropriate tests. If there are changes to the iOS app, add integration tests that hit the test backend. If there are just changes to the Rust libraries or binaries, then add tests there.
6. Update the make test target in the root project directory Makefile to call the new tests.
7. Run make test and esure all tests pass
8. Prepare a concise PR description

Remember we follow the schema-first design pattern. All messages to and from APIs are defined in the schemas dir. And we use rustls, not openssl in our containers.
