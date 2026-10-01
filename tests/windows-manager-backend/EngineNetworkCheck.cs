using System.Net;
using System.Net.NetworkInformation;
using NInfer.Manager;

internal static class EngineNetworkCheck
{
    public static void Run(Action<bool, string> check)
    {
        LanInterface Nic(string id, string address, bool gateway = false, bool up = true, bool virtualAdapter = false,
            NetworkInterfaceType type = NetworkInterfaceType.Ethernet) => new(id, type, up, virtualAdapter, gateway, [IPAddress.Parse(address)]);
        var network = new[] {
            Nic("virtual", "172.20.0.1", true, virtualAdapter: true),
            Nic("offline", "192.168.2.3", true, up: false),
            Nic("linklocal", "169.254.10.2", true),
            Nic("loopback", "127.0.0.1", true),
            Nic("ipv6", "fe80::1234", true),
            Nic("tunnel", "10.2.0.1", true, type: NetworkInterfaceType.Tunnel),
            Nic("secondary", "192.168.10.2"),
            Nic("wifi", "192.168.1.9", true, type: NetworkInterfaceType.Wireless80211)
        };
        check(EngineNetwork.SelectLanAddress(network)?.ToString() == "192.168.1.9", "LAN address chooses active gateway adapter and excludes software, disconnected, loopback, link-local and IPv6 candidates");
        check(EngineNetwork.SelectLanAddress(network.Reverse())?.ToString() == "192.168.1.9", "LAN selection does not depend on network enumeration order");
        check(EngineNetwork.SelectLanAddress(network.Take(6)) is null, "no usable LAN does not invent a loopback or wildcard access address");
        check(EngineNetwork.SelectLanAddress([Nic("direct", "192.168.2.9")])?.ToString() == "192.168.2.9", "direct Ethernet LAN works without an internet gateway");
        check(EngineNetwork.ApiBase("127.0.0.1", 18081) == "http://127.0.0.1:18081/v1", "local-only API address remains loopback");
    }
}
