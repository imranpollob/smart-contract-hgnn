// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Withdraw {
    mapping(address => uint256) public balance;

    function deposit() public payable {
        balance[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balance[msg.sender] >= amount);
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        balance[msg.sender] -= amount;
    }
}
