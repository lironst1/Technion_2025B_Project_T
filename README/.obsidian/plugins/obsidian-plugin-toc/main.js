var J = Object.create, d = Object.defineProperty, z = Object.getPrototypeOf, Z = Object.prototype.hasOwnProperty,
    K = Object.getOwnPropertyNames, Q = Object.getOwnPropertyDescriptor;
var T = u => d(u, "__esModule", {value: !0});
var B = (u, D) => () => (D || (D = {exports: {}}, u(D.exports, D)), D.exports), X = (u, D) => {
    for (var F in D) d(u, F, {get: D[F], enumerable: !0})
}, Y = (u, D, F) => {
    if (D && typeof D == "object" || typeof D == "function") for (let C of K(D)) !Z.call(u, C) && C !== "default" && d(u, C, {
        get: () => D[C],
        enumerable: !(F = Q(D, C)) || F.enumerable
    });
    return u
}, p = u => Y(T(d(u != null ? J(z(u)) : {}, "default", u && u.__esModule && "default" in u ? {
    get: () => u.default,
    enumerable: !0
} : {value: u, enumerable: !0})), u);
var k = B((cD, f) => {
    "use strict";

    function DD(u) {
        var D = void 0;
        typeof u == "string" ? D = [u] : D = u.raw;
        for (var F = "", C = 0; C < D.length; C++) F += D[C].replace(/\\\n[ \t]*/g, "").replace(/\\`/g, "`"), C < (arguments.length <= 1 ? 0 : arguments.length - 1) && (F += arguments.length <= C + 1 ? void 0 : arguments[C + 1]);
        var e = F.split(`
`), t = null;
        return e.forEach(function (E) {
            var s = E.match(/^(\s+)\S+/);
            if (s) {
                var n = s[1].length;
                t ? t = Math.min(t, n) : t = n
            }
        }), t !== null && (F = e.map(function (E) {
            return E[0] === " " ? E.slice(t) : E
        }).join(`
`)), F = F.trim(), F.replace(/\\n/g, `
`)
    }

    typeof f != "undefined" && (f.exports = DD)
});
var S = B((gD, y) => {
    y.exports = u => u != null && typeof u == "object" && u.constructor !== RegExp
});
var M = B((hD, L) => {
    "use strict";

    function m(u) {
        if (!(this instanceof m)) return new m(u);
        this.err = null, this.value = null;
        try {
            this.value = JSON.parse(u)
        } catch (D) {
            this.err = D
        }
    }

    L.exports = m
});
var _ = B(g => {
    "use strict";
    var A = g && g.__importDefault || function (u) {
        return u && u.__esModule ? u : {default: u}
    };
    Object.defineProperty(g, "__esModule", {value: !0});
    var uD = A(k()), FD = A(S()), P = A(M()), h = "twhZNwxI1aFG3r4";

    function O(u, ...D) {
        let F = "";
        for (let C = 0; C < u.length; C++) if (F += u[C], C < D.length) {
            let e = D[C], t = !1;
            if (P.default(e).value && (e = P.default(e).value, t = !0), e && e[h] || t) {
                let E = F.split(`
`), s = E[E.length - 1].search(/\S/), n = s > 0 ? " ".repeat(s) : "";
                (t ? JSON.stringify(e, null, 2) : e[h]).split(`
`).forEach((i, o) => {
                    o > 0 ? F += `
` + n + i : F += i
                })
            } else if (typeof e == "string" && e.includes(`
`)) {
                let E = F.match(/(?:^|\n)( *)$/);
                if (typeof e == "string") {
                    let s = E ? E[1] : "";
                    F += e.split(`
`).map((n, a) => (n = h + n, a === 0 ? n : `${s}${n}`)).join(`
`)
                } else F += e
            } else F += e
        }
        return F = uD.default(F), F.split(h).join("")
    }

    O.pretty = u => FD.default(u) ? {[h]: JSON.stringify(u, null, 2)} : u;
    g.default = O
});
var H = B((pD, x) => {
    x.exports = () => /(?:[#*0-9]\uFE0F?\u20E3|[\xA9\xAE\u203C\u2049\u2122\u2139\u2194-\u2199\u21A9\u21AA\u231A\u231B\u2328\u23CF\u23ED-\u23EF\u23F1\u23F2\u23F8-\u23FA\u24C2\u25AA\u25AB\u25B6\u25C0\u25FB\u25FC\u25FE\u2600-\u2604\u260E\u2611\u2614\u2615\u2618\u2620\u2622\u2623\u2626\u262A\u262E\u262F\u2638-\u263A\u2640\u2642\u2648-\u2653\u265F\u2660\u2663\u2665\u2666\u2668\u267B\u267E\u267F\u2692\u2694-\u2697\u2699\u269B\u269C\u26A0\u26A7\u26AA\u26B0\u26B1\u26BD\u26BE\u26C4\u26C8\u26CF\u26D1\u26D3\u26E9\u26F0-\u26F5\u26F7\u26F8\u26FA\u2702\u2708\u2709\u270F\u2712\u2714\u2716\u271D\u2721\u2733\u2734\u2744\u2747\u2757\u2763\u27A1\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B55\u3030\u303D\u3297\u3299]\uFE0F?|[\u261D\u270C\u270D](?:\uFE0F|\uD83C[\uDFFB-\uDFFF])?|[\u270A\u270B](?:\uD83C[\uDFFB-\uDFFF])?|[\u23E9-\u23EC\u23F0\u23F3\u25FD\u2693\u26A1\u26AB\u26C5\u26CE\u26D4\u26EA\u26FD\u2705\u2728\u274C\u274E\u2753-\u2755\u2795-\u2797\u27B0\u27BF\u2B50]|\u26F9(?:\uFE0F|\uD83C[\uDFFB-\uDFFF])?(?:\u200D[\u2640\u2642]\uFE0F?)?|\u2764\uFE0F?(?:\u200D(?:\uD83D\uDD25|\uD83E\uDE79))?|\uD83C(?:[\uDC04\uDD70\uDD71\uDD7E\uDD7F\uDE02\uDE37\uDF21\uDF24-\uDF2C\uDF36\uDF7D\uDF96\uDF97\uDF99-\uDF9B\uDF9E\uDF9F\uDFCD\uDFCE\uDFD4-\uDFDF\uDFF5\uDFF7]\uFE0F?|[\uDF85\uDFC2\uDFC7](?:\uD83C[\uDFFB-\uDFFF])?|[\uDFC3\uDFC4\uDFCA](?:\uD83C[\uDFFB-\uDFFF])?(?:\u200D[\u2640\u2642]\uFE0F?)?|[\uDFCB\uDFCC](?:\uFE0F|\uD83C[\uDFFB-\uDFFF])?(?:\u200D[\u2640\u2642]\uFE0F?)?|[\uDCCF\uDD8E\uDD91-\uDD9A\uDE01\uDE1A\uDE2F\uDE32-\uDE36\uDE38-\uDE3A\uDE50\uDE51\uDF00-\uDF20\uDF2D-\uDF35\uDF37-\uDF7C\uDF7E-\uDF84\uDF86-\uDF93\uDFA0-\uDFC1\uDFC5\uDFC6\uDFC8\uDFC9\uDFCF-\uDFD3\uDFE0-\uDFF0\uDFF8-\uDFFF]|\uDDE6\uD83C[\uDDE8-\uDDEC\uDDEE\uDDF1\uDDF2\uDDF4\uDDF6-\uDDFA\uDDFC\uDDFD\uDDFF]|\uDDE7\uD83C[\uDDE6\uDDE7\uDDE9-\uDDEF\uDDF1-\uDDF4\uDDF6-\uDDF9\uDDFB\uDDFC\uDDFE\uDDFF]|\uDDE8\uD83C[\uDDE6\uDDE8\uDDE9\uDDEB-\uDDEE\uDDF0-\uDDF5\uDDF7\uDDFA-\uDDFF]|\uDDE9\uD83C[\uDDEA\uDDEC\uDDEF\uDDF0\uDDF2\uDDF4\uDDFF]|\uDDEA\uD83C[\uDDE6\uDDE8\uDDEA\uDDEC\uDDED\uDDF7-\uDDFA]|\uDDEB\uD83C[\uDDEE-\uDDF0\uDDF2\uDDF4\uDDF7]|\uDDEC\uD83C[\uDDE6\uDDE7\uDDE9-\uDDEE\uDDF1-\uDDF3\uDDF5-\uDDFA\uDDFC\uDDFE]|\uDDED\uD83C[\uDDF0\uDDF2\uDDF3\uDDF7\uDDF9\uDDFA]|\uDDEE\uD83C[\uDDE8-\uDDEA\uDDF1-\uDDF4\uDDF6-\uDDF9]|\uDDEF\uD83C[\uDDEA\uDDF2\uDDF4\uDDF5]|\uDDF0\uD83C[\uDDEA\uDDEC-\uDDEE\uDDF2\uDDF3\uDDF5\uDDF7\uDDFC\uDDFE\uDDFF]|\uDDF1\uD83C[\uDDE6-\uDDE8\uDDEE\uDDF0\uDDF7-\uDDFB\uDDFE]|\uDDF2\uD83C[\uDDE6\uDDE8-\uDDED\uDDF0-\uDDFF]|\uDDF3\uD83C[\uDDE6\uDDE8\uDDEA-\uDDEC\uDDEE\uDDF1\uDDF4\uDDF5\uDDF7\uDDFA\uDDFF]|\uDDF4\uD83C\uDDF2|\uDDF5\uD83C[\uDDE6\uDDEA-\uDDED\uDDF0-\uDDF3\uDDF7-\uDDF9\uDDFC\uDDFE]|\uDDF6\uD83C\uDDE6|\uDDF7\uD83C[\uDDEA\uDDF4\uDDF8\uDDFA\uDDFC]|\uDDF8\uD83C[\uDDE6-\uDDEA\uDDEC-\uDDF4\uDDF7-\uDDF9\uDDFB\uDDFD-\uDDFF]|\uDDF9\uD83C[\uDDE6\uDDE8\uDDE9\uDDEB-\uDDED\uDDEF-\uDDF4\uDDF7\uDDF9\uDDFB\uDDFC\uDDFF]|\uDDFA\uD83C[\uDDE6\uDDEC\uDDF2\uDDF3\uDDF8\uDDFE\uDDFF]|\uDDFB\uD83C[\uDDE6\uDDE8\uDDEA\uDDEC\uDDEE\uDDF3\uDDFA]|\uDDFC\uD83C[\uDDEB\uDDF8]|\uDDFD\uD83C\uDDF0|\uDDFE\uD83C[\uDDEA\uDDF9]|\uDDFF\uD83C[\uDDE6\uDDF2\uDDFC]|\uDFF3\uFE0F?(?:\u200D(?:\u26A7\uFE0F?|\uD83C\uDF08))?|\uDFF4(?:\u200D\u2620\uFE0F?|\uDB40\uDC67\uDB40\uDC62\uDB40(?:\uDC65\uDB40\uDC6E\uDB40\uDC67|\uDC73\uDB40\uDC63\uDB40\uDC74|\uDC77\uDB40\uDC6C\uDB40\uDC73)\uDB40\uDC7F)?)|\uD83D(?:[\uDC3F\uDCFD\uDD49\uDD4A\uDD6F\uDD70\uDD73\uDD76-\uDD79\uDD87\uDD8A-\uDD8D\uDDA5\uDDA8\uDDB1\uDDB2\uDDBC\uDDC2-\uDDC4\uDDD1-\uDDD3\uDDDC-\uDDDE\uDDE1\uDDE3\uDDE8\uDDEF\uDDF3\uDDFA\uDECB\uDECD-\uDECF\uDEE0-\uDEE5\uDEE9\uDEF0\uDEF3]\uFE0F?|[\uDC42\uDC43\uDC46-\uDC50\uDC66\uDC67\uDC6B-\uDC6D\uDC72\uDC74-\uDC76\uDC78\uDC7C\uDC83\uDC85\uDC8F\uDC91\uDCAA\uDD7A\uDD95\uDD96\uDE4C\uDE4F\uDEC0\uDECC](?:\uD83C[\uDFFB-\uDFFF])?|[\uDC6E\uDC70\uDC71\uDC73\uDC77\uDC81\uDC82\uDC86\uDC87\uDE45-\uDE47\uDE4B\uDE4D\uDE4E\uDEA3\uDEB4-\uDEB6](?:\uD83C[\uDFFB-\uDFFF])?(?:\u200D[\u2640\u2642]\uFE0F?)?|[\uDD74\uDD90](?:\uFE0F|\uD83C[\uDFFB-\uDFFF])?|[\uDC00-\uDC07\uDC09-\uDC14\uDC16-\uDC3A\uDC3C-\uDC3E\uDC40\uDC44\uDC45\uDC51-\uDC65\uDC6A\uDC79-\uDC7B\uDC7D-\uDC80\uDC84\uDC88-\uDC8E\uDC90\uDC92-\uDCA9\uDCAB-\uDCFC\uDCFF-\uDD3D\uDD4B-\uDD4E\uDD50-\uDD67\uDDA4\uDDFB-\uDE2D\uDE2F-\uDE34\uDE37-\uDE44\uDE48-\uDE4A\uDE80-\uDEA2\uDEA4-\uDEB3\uDEB7-\uDEBF\uDEC1-\uDEC5\uDED0-\uDED2\uDED5-\uDED7\uDEDD-\uDEDF\uDEEB\uDEEC\uDEF4-\uDEFC\uDFE0-\uDFEB\uDFF0]|\uDC08(?:\u200D\u2B1B)?|\uDC15(?:\u200D\uD83E\uDDBA)?|\uDC3B(?:\u200D\u2744\uFE0F?)?|\uDC41\uFE0F?(?:\u200D\uD83D\uDDE8\uFE0F?)?|\uDC68(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:\uDC8B\u200D\uD83D)?\uDC68|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D(?:[\uDC68\uDC69]\u200D\uD83D(?:\uDC66(?:\u200D\uD83D\uDC66)?|\uDC67(?:\u200D\uD83D[\uDC66\uDC67])?)|[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uDC66(?:\u200D\uD83D\uDC66)?|\uDC67(?:\u200D\uD83D[\uDC66\uDC67])?)|\uD83E[\uDDAF-\uDDB3\uDDBC\uDDBD])|\uD83C(?:\uDFFB(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:\uDC8B\u200D\uD83D)?\uDC68\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D\uDC68\uD83C[\uDFFC-\uDFFF])))?|\uDFFC(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:\uDC8B\u200D\uD83D)?\uDC68\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D\uDC68\uD83C[\uDFFB\uDFFD-\uDFFF])))?|\uDFFD(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:\uDC8B\u200D\uD83D)?\uDC68\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D\uDC68\uD83C[\uDFFB\uDFFC\uDFFE\uDFFF])))?|\uDFFE(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:\uDC8B\u200D\uD83D)?\uDC68\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D\uDC68\uD83C[\uDFFB-\uDFFD\uDFFF])))?|\uDFFF(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:\uDC8B\u200D\uD83D)?\uDC68\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D\uDC68\uD83C[\uDFFB-\uDFFE])))?))?|\uDC69(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:\uDC8B\u200D\uD83D)?[\uDC68\uDC69]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D(?:[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uDC66(?:\u200D\uD83D\uDC66)?|\uDC67(?:\u200D\uD83D[\uDC66\uDC67])?|\uDC69\u200D\uD83D(?:\uDC66(?:\u200D\uD83D\uDC66)?|\uDC67(?:\u200D\uD83D[\uDC66\uDC67])?))|\uD83E[\uDDAF-\uDDB3\uDDBC\uDDBD])|\uD83C(?:\uDFFB(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:[\uDC68\uDC69]|\uDC8B\u200D\uD83D[\uDC68\uDC69])\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D[\uDC68\uDC69]\uD83C[\uDFFC-\uDFFF])))?|\uDFFC(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:[\uDC68\uDC69]|\uDC8B\u200D\uD83D[\uDC68\uDC69])\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D[\uDC68\uDC69]\uD83C[\uDFFB\uDFFD-\uDFFF])))?|\uDFFD(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:[\uDC68\uDC69]|\uDC8B\u200D\uD83D[\uDC68\uDC69])\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D[\uDC68\uDC69]\uD83C[\uDFFB\uDFFC\uDFFE\uDFFF])))?|\uDFFE(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:[\uDC68\uDC69]|\uDC8B\u200D\uD83D[\uDC68\uDC69])\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D[\uDC68\uDC69]\uD83C[\uDFFB-\uDFFD\uDFFF])))?|\uDFFF(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D\uD83D(?:[\uDC68\uDC69]|\uDC8B\u200D\uD83D[\uDC68\uDC69])\uD83C[\uDFFB-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83D[\uDC68\uDC69]\uD83C[\uDFFB-\uDFFE])))?))?|\uDC6F(?:\u200D[\u2640\u2642]\uFE0F?)?|\uDD75(?:\uFE0F|\uD83C[\uDFFB-\uDFFF])?(?:\u200D[\u2640\u2642]\uFE0F?)?|\uDE2E(?:\u200D\uD83D\uDCA8)?|\uDE35(?:\u200D\uD83D\uDCAB)?|\uDE36(?:\u200D\uD83C\uDF2B\uFE0F?)?)|\uD83E(?:[\uDD0C\uDD0F\uDD18-\uDD1F\uDD30-\uDD34\uDD36\uDD77\uDDB5\uDDB6\uDDBB\uDDD2\uDDD3\uDDD5\uDEC3-\uDEC5\uDEF0\uDEF2-\uDEF6](?:\uD83C[\uDFFB-\uDFFF])?|[\uDD26\uDD35\uDD37-\uDD39\uDD3D\uDD3E\uDDB8\uDDB9\uDDCD-\uDDCF\uDDD4\uDDD6-\uDDDD](?:\uD83C[\uDFFB-\uDFFF])?(?:\u200D[\u2640\u2642]\uFE0F?)?|[\uDDDE\uDDDF](?:\u200D[\u2640\u2642]\uFE0F?)?|[\uDD0D\uDD0E\uDD10-\uDD17\uDD20-\uDD25\uDD27-\uDD2F\uDD3A\uDD3F-\uDD45\uDD47-\uDD76\uDD78-\uDDB4\uDDB7\uDDBA\uDDBC-\uDDCC\uDDD0\uDDE0-\uDDFF\uDE70-\uDE74\uDE78-\uDE7C\uDE80-\uDE86\uDE90-\uDEAC\uDEB0-\uDEBA\uDEC0-\uDEC2\uDED0-\uDED9\uDEE0-\uDEE7]|\uDD3C(?:\u200D[\u2640\u2642]\uFE0F?|\uD83C[\uDFFB-\uDFFF])?|\uDDD1(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\uD83C[\uDF3E\uDF73\uDF7C\uDF84\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83E\uDDD1))|\uD83C(?:\uDFFB(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D(?:\uD83D\uDC8B\u200D)?\uD83E\uDDD1\uD83C[\uDFFC-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF84\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83E\uDDD1\uD83C[\uDFFB-\uDFFF])))?|\uDFFC(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D(?:\uD83D\uDC8B\u200D)?\uD83E\uDDD1\uD83C[\uDFFB\uDFFD-\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF84\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83E\uDDD1\uD83C[\uDFFB-\uDFFF])))?|\uDFFD(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D(?:\uD83D\uDC8B\u200D)?\uD83E\uDDD1\uD83C[\uDFFB\uDFFC\uDFFE\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF84\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83E\uDDD1\uD83C[\uDFFB-\uDFFF])))?|\uDFFE(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D(?:\uD83D\uDC8B\u200D)?\uD83E\uDDD1\uD83C[\uDFFB-\uDFFD\uDFFF]|\uD83C[\uDF3E\uDF73\uDF7C\uDF84\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83E\uDDD1\uD83C[\uDFFB-\uDFFF])))?|\uDFFF(?:\u200D(?:[\u2695\u2696\u2708]\uFE0F?|\u2764\uFE0F?\u200D(?:\uD83D\uDC8B\u200D)?\uD83E\uDDD1\uD83C[\uDFFB-\uDFFE]|\uD83C[\uDF3E\uDF73\uDF7C\uDF84\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\uD83E(?:[\uDDAF-\uDDB3\uDDBC\uDDBD]|\uDD1D\u200D\uD83E\uDDD1\uD83C[\uDFFB-\uDFFF])))?))?|\uDEF1(?:\uD83C(?:\uDFFB(?:\u200D\uD83E\uDEF2\uD83C[\uDFFC-\uDFFF])?|\uDFFC(?:\u200D\uD83E\uDEF2\uD83C[\uDFFB\uDFFD-\uDFFF])?|\uDFFD(?:\u200D\uD83E\uDEF2\uD83C[\uDFFB\uDFFC\uDFFE\uDFFF])?|\uDFFE(?:\u200D\uD83E\uDEF2\uD83C[\uDFFB-\uDFFD\uDFFF])?|\uDFFF(?:\u200D\uD83E\uDEF2\uD83C[\uDFFB-\uDFFE])?))?))/g
});
var j = B((fD, I) => {
    I.exports = function (u, D) {
        D = D || {}, D.listUnicodeChar = D.hasOwnProperty("listUnicodeChar") ? D.listUnicodeChar : !1, D.stripListLeaders = D.hasOwnProperty("stripListLeaders") ? D.stripListLeaders : !0, D.gfm = D.hasOwnProperty("gfm") ? D.gfm : !0, D.useImgAltText = D.hasOwnProperty("useImgAltText") ? D.useImgAltText : !0, D.abbr = D.hasOwnProperty("abbr") ? D.abbr : !1, D.replaceLinksWithURL = D.hasOwnProperty("replaceLinksWithURL") ? D.replaceLinksWithURL : !1, D.htmlTagsToSkip = D.hasOwnProperty("htmlTagsToSkip") ? D.htmlTagsToSkip : [];
        var F = u || "";
        F = F.replace(/^(-\s*?|\*\s*?|_\s*?){3,}\s*/gm, "");
        try {
            D.stripListLeaders && (D.listUnicodeChar ? F = F.replace(/^([\s\t]*)([\*\-\+]|\d+\.)\s+/gm, D.listUnicodeChar + " $1") : F = F.replace(/^([\s\t]*)([\*\-\+]|\d+\.)\s+/gm, "$1")), D.gfm && (F = F.replace(/\n={2,}/g, `
`).replace(/~{3}.*\n/g, "").replace(/~~/g, "").replace(/`{3}.*\n/g, "")), D.abbr && (F = F.replace(/\*\[.*\]:.*\n/, "")), F = F.replace(/<[^>]*>/g, "");
            var C = new RegExp("<[^>]*>", "g");
            if (D.htmlTagsToSkip.length > 0) {
                var e = "(?!" + D.htmlTagsToSkip.join("|") + ")";
                C = new RegExp("<" + e + "[^>]*>", "ig")
            }
            F = F.replace(C, "").replace(/^[=\-]{2,}\s*$/g, "").replace(/\[\^.+?\](\: .*?$)?/g, "").replace(/\s{0,2}\[.*?\]: .*?$/g, "").replace(/\!\[(.*?)\][\[\(].*?[\]\)]/g, D.useImgAltText ? "$1" : "").replace(/\[([^\]]*?)\][\[\(].*?[\]\)]/g, D.replaceLinksWithURL ? "$2" : "$1").replace(/^\s{0,3}>\s?/gm, "").replace(/^\s{1,2}\[(.*?)\]: (\S+)( ".*?")?\s*$/g, "").replace(/^(\n)?\s{0,}#{1,6}\s+| {0,}(\n)?\s{0,}#{0,} #{0,}(\n)?\s{0,}$/gm, "$1$2$3").replace(/([\*]+)(\S)(.*?\S)??\1/g, "$2$3").replace(/(^|\W)([_]+)(\S)(.*?\S)??\2($|\W)/g, "$1$3$4$5").replace(/(`{3,})(.*?)\1/gm, "$2").replace(/`(.+?)`/g, "$1").replace(/~(.*?)~/g, "$1")
        } catch (t) {
            return console.error(t), u
        }
        return F
    }
});
var R = B((mD, N) => {
    "use strict";
    var eD = H(), tD = j();

    function CD(u, D) {
        return u = u.replace(/[^a-z0-9]+/g, "_"), u = u.replace(/^_+|_+$/, ""), u = u.replace(/^([^a-z])/, "_$1"), D && (u += "_" + D), u
    }

    function U(u) {
        return u.replace(/ /g, "-").replace(/%([abcdef]|\d){2,2}/ig, "").replace(/[\/?!:\[\]`.,()*"';{}+=<>~\$|#@&–—]/g, "").replace(/[。？！，、；：“”【】（）〔〕［］﹃﹄“ ”‘’﹁﹂—…－～《》〈〉「」]/g, "")
    }

    function nD(u, D) {
        return u = U(u), D && (u += "-" + D), u = u.replace(eD(), ""), u = tD(u), u
    }

    function ED(u, D) {
        return u = "markdown-header-" + U(u), u = u.replace(/--+/g, "-"), D && (u += "_" + D), u
    }

    function aD(u) {
        return u.replace(/ /g, "").replace(/[\/?:\[\]`.,()*"';{}\-+=<>!@#%^&\\\|]/g, "").replace(/\$/g, "d").replace(/~/g, "t")
    }

    function rD(u) {
        return u = aD(u), u
    }

    function iD(u, D) {
        return u = u.replace(/<(.*)>(.*)<\/\1>/g, "$2").replace(/!\[.*\]\(.*\)/g, "").replace(/\[(.*)\]\(.*\)/, "$1").replace(/\s+/g, "-").replace(/[\/?!:\[\]`.,()*"';{}+=<>~\$|#@]/g, "").replace(/[。？！，、；：“”【】（）〔〕［］﹃﹄“ ”‘’﹁﹂—…－～《》〈〉「」]/g, "").replace(/[-]+/g, "-").replace(/^-/, "").replace(/-$/, ""), D && (u += "-" + D), u
    }

    N.exports = function (D, F, C, e) {
        F = F || "github.com";
        var t, E = encodeURI;
        switch (F) {
            case"github.com":
                t = nD, E = function (a) {
                    var l = encodeURI(a);
                    return l.replace(/%E2%80%8D/g, "\u200D")
                };
                break;
            case"bitbucket.org":
                t = ED;
                break;
            case"gitlab.com":
                t = iD;
                break;
            case"nodejs.org":
                if (!e) throw new Error("Need module name to generate proper anchor for " + F);
                t = function (a, l) {
                    return CD(e + "." + a, l)
                };
                break;
            case"ghost.org":
                t = rD;
                break;
            default:
                throw new Error("Unknown mode: " + F)
        }

        function s(a) {
            for (var l = "", i = 0; i < a.length; ++i) a[i] >= "A" && a[i] <= "Z" ? l += a[i].toLowerCase() : l += a[i];
            return l
        }

        var n = t(s(D.trim()), C);
        return "[" + D + "](#" + E(n) + ")"
    }
});
T(exports);
X(exports, {default: () => BD});
var r = p(require("obsidian"));
var b = p(_()), G = p(require("obsidian")), V = p(R()), w = (u, D) => {
    let F = u.filter(C => C.position.end.line < D.line);
    return F.length ? F[F.length - 1].level : 0
}, sD = (u, D) => u.filter(F => F.position.end.line > D.line), lD = (u, D) => {
    let F = u.indexOf(D);
    return u.slice(0, F).reverse().find((e, t, E) => e.level == D.level - 1)
}, q = ({headings: u = []}, D, F) => {
    let C = w(u, D), e = sD(u, D), t = [];
    for (let n of e) {
        if (n.level <= C) break;
        n.level >= F.minimumDepth && n.level <= F.maximumDepth && t.push(n)
    }
    if (!t.length) {
        new G.Notice(b.default`
        No headings below cursor matched settings 
        (min: ${F.minimumDepth}) (max: ${F.maximumDepth})
      `);
        return
    }
    let E = t[0].level, s = t.map(n => {
        let a = F.listStyle === "number" && "1." || "-", l = new Array(Math.max(0, n.level - E)).fill("	").join(""),
            i = lD(t, n), o = `${l}${a}`, $ = n.heading, c;
        return F.useMarkdown && F.githubCompat ? `${o} ${(0, V.default)(n.heading)}` : (F.useMarkdown ? c = encodeURI(n.heading) : typeof i == "undefined" ? c = n.heading : c = `${i.heading}#${n.heading}`, F.useMarkdown ? `${o} [${$}](#${c})` : `${o} [[#${c}|${$}]]`)
    });
    return b.default`
    ${F.title ? `${F.title}
` : ""}
    ${`${s.join(`
`)}
`}
  `
};
var W = class extends r.PluginSettingTab {
    constructor(D, F) {
        super(D, F);
        this.plugin = F
    }

    display() {
        let {containerEl: D} = this;
        D.empty(), D.createEl("h2", {text: "Table of Contents - Settings"}), new r.Setting(D).setName("List Style").setDesc("The type of list to render the table of contents as.").addDropdown(e => e.setValue(this.plugin.settings.listStyle).addOption("bullet", "Bullet").addOption("number", "Number").onChange(t => {
            this.plugin.settings.listStyle = t, this.plugin.saveData(this.plugin.settings), this.display()
        })), new r.Setting(D).setName("Title").setDesc("Optional title to put before the table of contents").addText(e => e.setPlaceholder("**Table of Contents**").setValue(this.plugin.settings.title || "").onChange(t => {
            this.plugin.settings.title = t, this.plugin.saveData(this.plugin.settings)
        })), new r.Setting(D).setName("Minimum Header Depth").setDesc("The lowest header depth to add to the table of contents. Defaults to 2").addSlider(e => e.setValue(this.plugin.settings.minimumDepth).setDynamicTooltip().setLimits(1, 6, 1).onChange(t => {
            this.plugin.settings.minimumDepth = t, this.plugin.saveData(this.plugin.settings)
        })), new r.Setting(D).setName("Maximum Header Depth").setDesc("The highest header depth to add to the table of contents. Defaults to 6").addSlider(e => e.setValue(this.plugin.settings.maximumDepth).setDynamicTooltip().setLimits(1, 6, 1).onChange(t => {
            this.plugin.settings.maximumDepth = t, this.plugin.saveData(this.plugin.settings)
        })), new r.Setting(D).setName("Use Markdown links").setDesc("Auto-generate Markdown links, instead of the default WikiLinks").addToggle(e => e.setValue(this.plugin.settings.useMarkdown).onChange(t => {
            this.plugin.settings.useMarkdown = t, this.plugin.saveData(this.plugin.settings), t || C.components[0].setValue(!1), C.setDisabled(!t)
        }));
        let F = new DocumentFragment;
        F.appendText("Github generates section links differently than Obsidian, this setting uses "), F.createEl("a", {
            href: "https://github.com/thlorenz/anchor-markdown-header",
            text: "anchor-markdown-header"
        }), F.appendText(" to generate the proper links.");
        let C = new r.Setting(D).setName("Github compliant Markdown section links").setDesc(F).setDisabled(!this.plugin.settings.useMarkdown).addToggle(e => e.setValue(this.plugin.settings.githubCompat ?? !1).setDisabled(!this.plugin.settings.useMarkdown).onChange(t => {
            this.plugin.settings.githubCompat = t, this.plugin.saveData(this.plugin.settings)
        }))
    }
}, v = class extends r.Plugin {
    constructor() {
        super(...arguments);
        this.settings = {minimumDepth: 2, maximumDepth: 6, listStyle: "bullet", useMarkdown: !1};
        this.createTocForActiveFile = (D = this.settings) => () => {
            let F = this.app.workspace.getActiveViewOfType(r.MarkdownView);
            if (F && F.file) {
                let C = F.sourceMode.cmEditor, e = C.getCursor(), t = this.app.metadataCache.getFileCache(F.file) || {},
                    E = q(t, e, typeof D == "function" ? D(t, e) : D);
                E && C.replaceRange(E, e)
            }
        }
    }

    async onload() {
        console.log("Load Table of Contents plugin"), this.settings = {...this.settings, ...await this.loadData()}, this.addCommand({
            id: "create-toc",
            name: "Create table of contents",
            callback: this.createTocForActiveFile()
        }), this.addCommand({
            id: "create-toc-next-level",
            name: "Create table of contents for next heading level",
            callback: this.createTocForActiveFile((D, F) => {
                let C = w(D.headings || [], F), e = Math.max(C + 1, this.settings.minimumDepth);
                return {...this.settings, minimumDepth: e, maximumDepth: e}
            })
        }), this.addSettingTab(new W(this.app, this))
    }
}, BD = v;

/* nosourcemap */