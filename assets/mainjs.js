window.sendDataToFlask = sendDataToFlask;

function is(o, t) {
    if (!(o instanceof t)) throw new TypeError("Cannot call a class as a function")
}

function jl(o, t) {
    for (var r = 0; r < t.length; r++) {
        var e = t[r];
        e.enumerable = e.enumerable || !1, e.configurable = !0, "value" in e && (e.writable = !0), Object.defineProperty(o, e.key, e)
    }
}

function rs(o, t, r) {
    return t && jl(o.prototype, t), o
}

function Gl(o, t, r) {
    return t in o ? Object.defineProperty(o, t, {
        value: r,
        enumerable: !0,
        configurable: !0,
        writable: !0
    }) : o[t] = r, o
}

function Co(o, t) {
    var r = Object.keys(o);
    if (Object.getOwnPropertySymbols) {
        var e = Object.getOwnPropertySymbols(o);
        t && (e = e.filter(function(i) {
            return Object.getOwnPropertyDescriptor(o, i).enumerable
        })), r.push.apply(r, e)
    }
    return r
}

function cs(o) {
    for (var t = 1; t < arguments.length; t++) {
        var r = arguments[t] != null ? arguments[t] : {};
        t % 2 ? Co(Object(r), !0).forEach(function(e) {
            Gl(o, e, r[e])
        }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(o, Object.getOwnPropertyDescriptors(r)) : Co(Object(r)).forEach(function(e) {
            Object.defineProperty(o, e, Object.getOwnPropertyDescriptor(r, e))
        })
    }
    return o
}

function fa(o, t) {
    if (typeof t != "function" && t !== null) throw new TypeError("Super expression must either be null or a function");
    o.prototype = Object.create(t && t.prototype, {
        constructor: {
            value: o,
            writable: !0,
            configurable: !0
        }
    }), t && Cs(o, t)
}

function Ve(o) {
    return Ve = Object.setPrototypeOf ? Object.getPrototypeOf : function(r) {
        return r.__proto__ || Object.getPrototypeOf(r)
    }, Ve(o)
}

function Cs(o, t) {
    return Cs = Object.setPrototypeOf || function(e, i) {
        return e.__proto__ = i, e
    }, Cs(o, t)
}

function Kl() {
    if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
    if (typeof Proxy == "function") return !0;
    try {
        return Date.prototype.toString.call(Reflect.construct(Date, [], function() {})), !0
    } catch {
        return !1
    }
}

function ha(o) {
    if (o === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return o
}

function Zl(o, t) {
    return t && (typeof t == "object" || typeof t == "function") ? t : ha(o)
}

function da(o) {
    var t = Kl();
    return function() {
        var e = Ve(o),
            i;
        if (t) {
            var n = Ve(this).constructor;
            i = Reflect.construct(e, arguments, n)
        } else i = e.apply(this, arguments);
        return Zl(this, i)
    }
}

function Ql(o, t) {
    for (; !Object.prototype.hasOwnProperty.call(o, t) && (o = Ve(o), o !== null););
    return o
}

function ci(o, t, r) {
    return typeof Reflect < "u" && Reflect.get ? ci = Reflect.get : ci = function(i, n, s) {
        var a = Ql(i, n);
        if (a) {
            var l = Object.getOwnPropertyDescriptor(a, n);
            return l.get ? l.get.call(s) : l.value
        }
    }, ci(o, t, r || o)
}

function dr(o, t) {
    return ec(o) || rc(o, t) || pa(o, t) || sc()
}

function Jl(o) {
    return tc(o) || ic(o) || pa(o) || nc()
}

function tc(o) {
    if (Array.isArray(o)) return Ps(o)
}

function ec(o) {
    if (Array.isArray(o)) return o
}

function ic(o) {
    if (typeof Symbol < "u" && Symbol.iterator in Object(o)) return Array.from(o)
}

function rc(o, t) {
    if (!(typeof Symbol > "u" || !(Symbol.iterator in Object(o)))) {
        var r = [],
            e = !0,
            i = !1,
            n = void 0;
        try {
            for (var s = o[Symbol.iterator](), a; !(e = (a = s.next()).done) && (r.push(a.value), !(t && r.length === t)); e = !0);
        } catch (l) {
            i = !0, n = l
        } finally {
            try {
                !e && s.return != null && s.return()
            } finally {
                if (i) throw n
            }
        }
        return r
    }
}

function pa(o, t) {
    if (o) {
        if (typeof o == "string") return Ps(o, t);
        var r = Object.prototype.toString.call(o).slice(8, -1);
        if (r === "Object" && o.constructor && (r = o.constructor.name), r === "Map" || r === "Set") return Array.from(o);
        if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Ps(o, t)
    }
}

function Ps(o, t) {
    (t == null || t > o.length) && (t = o.length);
    for (var r = 0, e = new Array(t); r < t; r++) e[r] = o[r];
    return e
}

function nc() {
    throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)
}

function sc() {
    throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)
}
var mr = {
        el: document,
        name: "scroll",
        offset: [0, 0],
        repeat: !1,
        smooth: !1,
        initPosition: {
            x: 0,
            y: 0
        },
        direction: "vertical",
        gestureDirection: "vertical",
        reloadOnContextChange: !1,
        lerp: .1,
        class: "is-inview",
        scrollbarContainer: !1,
        scrollbarClass: "c-scrollbar",
        scrollingClass: "has-scroll-scrolling",
        draggingClass: "has-scroll-dragging",
        smoothClass: "has-scroll-smooth",
        initClass: "has-scroll-init",
        getSpeed: !1,
        getDirection: !1,
        scrollFromAnywhere: !1,
        multiplier: 1,
        firefoxMultiplier: 50,
        touchMultiplier: 2,
        resetNativeScroll: !0,
        tablet: {
            smooth: !1,
            direction: "vertical",
            gestureDirection: "vertical",
            breakpoint: 1024
        },
        smartphone: {
            smooth: !1,
            direction: "vertical",
            gestureDirection: "vertical"
        }
    },
    _a = function() {
        function o() {
            var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
            is(this, o), Object.assign(this, mr, t), this.smartphone = mr.smartphone, t.smartphone && Object.assign(this.smartphone, t.smartphone), this.tablet = mr.tablet, t.tablet && Object.assign(this.tablet, t.tablet), this.namespace = "locomotive", this.html = document.documentElement, this.windowHeight = window.innerHeight, this.windowWidth = window.innerWidth, this.windowMiddle = {
                x: this.windowWidth / 2,
                y: this.windowHeight / 2
            }, this.els = {}, this.currentElements = {}, this.listeners = {}, this.hasScrollTicking = !1, this.hasCallEventSet = !1, this.checkScroll = this.checkScroll.bind(this), this.checkResize = this.checkResize.bind(this), this.checkEvent = this.checkEvent.bind(this), this.instance = {
                scroll: {
                    x: 0,
                    y: 0
                },
                limit: {
                    x: this.html.offsetWidth,
                    y: this.html.offsetHeight
                },
                currentElements: this.currentElements
            }, this.isMobile ? this.isTablet ? this.context = "tablet" : this.context = "smartphone" : this.context = "desktop", this.isMobile && (this.direction = this[this.context].direction), this.direction === "horizontal" ? this.directionAxis = "x" : this.directionAxis = "y", this.getDirection && (this.instance.direction = null), this.getDirection && (this.instance.speed = 0), this.html.classList.add(this.initClass), window.addEventListener("resize", this.checkResize, !1)
        }
        return rs(o, [{
            key: "init",
            value: function() {
                this.initEvents()
            }
        }, {
            key: "checkScroll",
            value: function() {
                this.dispatchScroll()
            }
        }, {
            key: "checkResize",
            value: function() {
                var r = this;
                this.resizeTick || (this.resizeTick = !0, requestAnimationFrame(function() {
                    r.resize(), r.resizeTick = !1
                }))
            }
        }, {
            key: "resize",
            value: function() {}
        }, {
            key: "checkContext",
            value: function() {
                if (this.reloadOnContextChange) {
                    this.isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1 || this.windowWidth < this.tablet.breakpoint, this.isTablet = this.isMobile && this.windowWidth >= this.tablet.breakpoint;
                    var r = this.context;
                    if (this.isMobile ? this.isTablet ? this.context = "tablet" : this.context = "smartphone" : this.context = "desktop", r != this.context) {
                        var e = r == "desktop" ? this.smooth : this[r].smooth,
                            i = this.context == "desktop" ? this.smooth : this[this.context].smooth;
                        e != i && window.location.reload()
                    }
                }
            }
        }, {
            key: "initEvents",
            value: function() {
                var r = this;
                this.scrollToEls = this.el.querySelectorAll("[data-".concat(this.name, "-to]")), this.setScrollTo = this.setScrollTo.bind(this), this.scrollToEls.forEach(function(e) {
                    e.addEventListener("click", r.setScrollTo, !1)
                })
            }
        }, {
            key: "setScrollTo",
            value: function(r) {
                r.preventDefault(), this.scrollTo(r.currentTarget.getAttribute("data-".concat(this.name, "-href")) || r.currentTarget.getAttribute("href"), {
                    offset: r.currentTarget.getAttribute("data-".concat(this.name, "-offset"))
                })
            }
        }, {
            key: "addElements",
            value: function() {}
        }, {
            key: "detectElements",
            value: function(r) {
                var e = this,
                    i = this.instance.scroll.y,
                    n = i + this.windowHeight,
                    s = this.instance.scroll.x,
                    a = s + this.windowWidth;
                Object.entries(this.els).forEach(function(l) {
                    var c = dr(l, 2),
                        u = c[0],
                        h = c[1];
                    if (h && (!h.inView || r) && (e.direction === "horizontal" ? a >= h.left && s < h.right && e.setInView(h, u) : n >= h.top && i < h.bottom && e.setInView(h, u)), h && h.inView)
                        if (e.direction === "horizontal") {
                            var d = h.right - h.left;
                            h.progress = (e.instance.scroll.x - (h.left - e.windowWidth)) / (d + e.windowWidth), (a < h.left || s > h.right) && e.setOutOfView(h, u)
                        } else {
                            var f = h.bottom - h.top;
                            h.progress = (e.instance.scroll.y - (h.top - e.windowHeight)) / (f + e.windowHeight), (n < h.top || i > h.bottom) && e.setOutOfView(h, u)
                        }
                }), this.hasScrollTicking = !1
            }
        }, {
            key: "setInView",
            value: function(r, e) {
                this.els[e].inView = !0, r.el.classList.add(r.class), this.currentElements[e] = r, r.call && this.hasCallEventSet && (this.dispatchCall(r, "enter"), r.repeat || (this.els[e].call = !1))
            }
        }, {
            key: "setOutOfView",
            value: function(r, e) {
                var i = this;
                this.els[e].inView = !1, Object.keys(this.currentElements).forEach(function(n) {
                    n === e && delete i.currentElements[n]
                }), r.call && this.hasCallEventSet && this.dispatchCall(r, "exit"), r.repeat && r.el.classList.remove(r.class)
            }
        }, {
            key: "dispatchCall",
            value: function(r, e) {
                this.callWay = e, this.callValue = r.call.split(",").map(function(n) {
                    return n.trim()
                }), this.callObj = r, this.callValue.length == 1 && (this.callValue = this.callValue[0]);
                var i = new Event(this.namespace + "call");
                this.el.dispatchEvent(i)
            }
        }, {
            key: "dispatchScroll",
            value: function() {
                var r = new Event(this.namespace + "scroll");
                this.el.dispatchEvent(r)
            }
        }, {
            key: "setEvents",
            value: function(r, e) {
                this.listeners[r] || (this.listeners[r] = []);
                var i = this.listeners[r];
                i.push(e), i.length === 1 && this.el.addEventListener(this.namespace + r, this.checkEvent, !1), r === "call" && (this.hasCallEventSet = !0, this.detectElements(!0))
            }
        }, {
            key: "unsetEvents",
            value: function(r, e) {
                if (this.listeners[r]) {
                    var i = this.listeners[r],
                        n = i.indexOf(e);
                    n < 0 || (i.splice(n, 1), i.index === 0 && this.el.removeEventListener(this.namespace + r, this.checkEvent, !1))
                }
            }
        }, {
            key: "checkEvent",
            value: function(r) {
                var e = this,
                    i = r.type.replace(this.namespace, ""),
                    n = this.listeners[i];
                !n || n.length === 0 || n.forEach(function(s) {
                    switch (i) {
                        case "scroll":
                            return s(e.instance);
                        case "call":
                            return s(e.callValue, e.callWay, e.callObj);
                        default:
                            return s()
                    }
                })
            }
        }, {
            key: "startScroll",
            value: function() {}
        }, {
            key: "stopScroll",
            value: function() {}
        }, {
            key: "setScroll",
            value: function(r, e) {
                this.instance.scroll = {
                    x: 0,
                    y: 0
                }
            }
        }, {
            key: "destroy",
            value: function() {
                var r = this;
                window.removeEventListener("resize", this.checkResize, !1), Object.keys(this.listeners).forEach(function(e) {
                    r.el.removeEventListener(r.namespace + e, r.checkEvent, !1)
                }), this.listeners = {}, this.scrollToEls.forEach(function(e) {
                    e.removeEventListener("click", r.setScrollTo, !1)
                }), this.html.classList.remove(this.initClass)
            }
        }]), o
    }(),
    oc = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};

function ga(o, t) {
    return t = {
        exports: {}
    }, o(t, t.exports), t.exports
}
var ma = ga(function(o, t) {
    (function() {
        function r() {
            var e = window,
                i = document;
            if ("scrollBehavior" in i.documentElement.style && e.__forceSmoothScrollPolyfill__ !== !0) return;
            var n = e.HTMLElement || e.Element,
                s = 468,
                a = {
                    scroll: e.scroll || e.scrollTo,
                    scrollBy: e.scrollBy,
                    elementScroll: n.prototype.scroll || h,
                    scrollIntoView: n.prototype.scrollIntoView
                },
                l = e.performance && e.performance.now ? e.performance.now.bind(e.performance) : Date.now;

            function c(g) {
                var y = ["MSIE ", "Trident/", "Edge/"];
                return new RegExp(y.join("|")).test(g)
            }
            var u = c(e.navigator.userAgent) ? 1 : 0;

            function h(g, y) {
                this.scrollLeft = g, this.scrollTop = y
            }

            function d(g) {
                return .5 * (1 - Math.cos(Math.PI * g))
            }

            function f(g) {
                if (g === null || typeof g != "object" || g.behavior === void 0 || g.behavior === "auto" || g.behavior === "instant") return !0;
                if (typeof g == "object" && g.behavior === "smooth") return !1;
                throw new TypeError("behavior member of ScrollOptions " + g.behavior + " is not a valid value for enumeration ScrollBehavior.")
            }

            function _(g, y) {
                if (y === "Y") return g.clientHeight + u < g.scrollHeight;
                if (y === "X") return g.clientWidth + u < g.scrollWidth
            }

            function p(g, y) {
                var S = e.getComputedStyle(g, null)["overflow" + y];
                return S === "auto" || S === "scroll"
            }

            function v(g) {
                var y = _(g, "Y") && p(g, "Y"),
                    S = _(g, "X") && p(g, "X");
                return y || S
            }

            function T(g) {
                for (; g !== i.body && v(g) === !1;) g = g.parentNode || g.host;
                return g
            }

            function b(g) {
                var y = l(),
                    S, w, C, k = (y - g.startTime) / s;
                k = k > 1 ? 1 : k, S = d(k), w = g.startX + (g.x - g.startX) * S, C = g.startY + (g.y - g.startY) * S, g.method.call(g.scrollable, w, C), (w !== g.x || C !== g.y) && e.requestAnimationFrame(b.bind(e, g))
            }

            function E(g, y, S) {
                var w, C, k, O, L = l();
                g === i.body ? (w = e, C = e.scrollX || e.pageXOffset, k = e.scrollY || e.pageYOffset, O = a.scroll) : (w = g, C = g.scrollLeft, k = g.scrollTop, O = h), b({
                    scrollable: w,
                    method: O,
                    startTime: L,
                    startX: C,
                    startY: k,
                    x: y,
                    y: S
                })
            }
            e.scroll = e.scrollTo = function() {
                if (arguments[0] !== void 0) {
                    if (f(arguments[0]) === !0) {
                        a.scroll.call(e, arguments[0].left !== void 0 ? arguments[0].left : typeof arguments[0] != "object" ? arguments[0] : e.scrollX || e.pageXOffset, arguments[0].top !== void 0 ? arguments[0].top : arguments[1] !== void 0 ? arguments[1] : e.scrollY || e.pageYOffset);
                        return
                    }
                    E.call(e, i.body, arguments[0].left !== void 0 ? ~~arguments[0].left : e.scrollX || e.pageXOffset, arguments[0].top !== void 0 ? ~~arguments[0].top : e.scrollY || e.pageYOffset)
                }
            }, e.scrollBy = function() {
                if (arguments[0] !== void 0) {
                    if (f(arguments[0])) {
                        a.scrollBy.call(e, arguments[0].left !== void 0 ? arguments[0].left : typeof arguments[0] != "object" ? arguments[0] : 0, arguments[0].top !== void 0 ? arguments[0].top : arguments[1] !== void 0 ? arguments[1] : 0);
                        return
                    }
                    E.call(e, i.body, ~~arguments[0].left + (e.scrollX || e.pageXOffset), ~~arguments[0].top + (e.scrollY || e.pageYOffset))
                }
            }, n.prototype.scroll = n.prototype.scrollTo = function() {
                if (arguments[0] !== void 0) {
                    if (f(arguments[0]) === !0) {
                        if (typeof arguments[0] == "number" && arguments[1] === void 0) throw new SyntaxError("Value could not be converted");
                        a.elementScroll.call(this, arguments[0].left !== void 0 ? ~~arguments[0].left : typeof arguments[0] != "object" ? ~~arguments[0] : this.scrollLeft, arguments[0].top !== void 0 ? ~~arguments[0].top : arguments[1] !== void 0 ? ~~arguments[1] : this.scrollTop);
                        return
                    }
                    var g = arguments[0].left,
                        y = arguments[0].top;
                    E.call(this, this, typeof g > "u" ? this.scrollLeft : ~~g, typeof y > "u" ? this.scrollTop : ~~y)
                }
            }, n.prototype.scrollBy = function() {
                if (arguments[0] !== void 0) {
                    if (f(arguments[0]) === !0) {
                        a.elementScroll.call(this, arguments[0].left !== void 0 ? ~~arguments[0].left + this.scrollLeft : ~~arguments[0] + this.scrollLeft, arguments[0].top !== void 0 ? ~~arguments[0].top + this.scrollTop : ~~arguments[1] + this.scrollTop);
                        return
                    }
                    this.scroll({
                        left: ~~arguments[0].left + this.scrollLeft,
                        top: ~~arguments[0].top + this.scrollTop,
                        behavior: arguments[0].behavior
                    })
                }
            }, n.prototype.scrollIntoView = function() {
                if (f(arguments[0]) === !0) {
                    a.scrollIntoView.call(this, arguments[0] === void 0 ? !0 : arguments[0]);
                    return
                }
                var g = T(this),
                    y = g.getBoundingClientRect(),
                    S = this.getBoundingClientRect();
                g !== i.body ? (E.call(this, g, g.scrollLeft + S.left - y.left, g.scrollTop + S.top - y.top), e.getComputedStyle(g).position !== "fixed" && e.scrollBy({
                    left: y.left,
                    top: y.top,
                    behavior: "smooth"
                })) : e.scrollBy({
                    left: S.left,
                    top: S.top,
                    behavior: "smooth"
                })
            }
        }
        o.exports = {
            polyfill: r
        }
    })()
});
ma.polyfill;
/*
var ac = function(o) {
    fa(r, o);
    var t = da(r);

    function r() {
        var e, i = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
        return is(this, r), e = t.call(this, i), e.resetNativeScroll && (history.scrollRestoration && (history.scrollRestoration = "manual"), window.scrollTo(0, 0)), window.addEventListener("scroll", e.checkScroll, !1), window.smoothscrollPolyfill === void 0 && (window.smoothscrollPolyfill = ma, window.smoothscrollPolyfill.polyfill()), e
    }
    return rs(r, [{
        key: "init",
        value: function() {
            this.instance.scroll.y = window.pageYOffset, this.addElements(), this.detectElements(), ci(Ve(r.prototype), "init", this).call(this)
        }
    }, {
        key: "checkScroll",
        value: function() {
            var i = this;
            ci(Ve(r.prototype), "checkScroll", this).call(this), this.getDirection && this.addDirection(), this.getSpeed && (this.addSpeed(), this.speedTs = Date.now()), this.instance.scroll.y = window.pageYOffset, Object.entries(this.els).length && (this.hasScrollTicking || (requestAnimationFrame(function() {
                i.detectElements()
            }), this.hasScrollTicking = !0))
        }
    }, {
        key: "addDirection",
        value: function() {
            window.pageYOffset > this.instance.scroll.y ? this.instance.direction !== "down" && (this.instance.direction = "down") : window.pageYOffset < this.instance.scroll.y && this.instance.direction !== "up" && (this.instance.direction = "up")
        }
    }, {
        key: "addSpeed",
        value: function() {
            window.pageYOffset != this.instance.scroll.y ? this.instance.speed = (window.pageYOffset - this.instance.scroll.y) / Math.max(1, Date.now() - this.speedTs) : this.instance.speed = 0
        }
    }, {
        key: "resize",
        value: function() {
            Object.entries(this.els).length && (this.windowHeight = window.innerHeight, this.updateElements())
        }
    }, {
        key: "addElements",
        value: function() {
            var i = this;
            this.els = {};
            var n = this.el.querySelectorAll("[data-" + this.name + "]");
            n.forEach(function(s, a) {
                s.getBoundingClientRect();
                var l = s.dataset[i.name + "Class"] || i.class,
                    c = typeof s.dataset[i.name + "Id"] == "string" ? s.dataset[i.name + "Id"] : a,
                    u, h, d = typeof s.dataset[i.name + "Offset"] == "string" ? s.dataset[i.name + "Offset"].split(",") : i.offset,
                    f = s.dataset[i.name + "Repeat"],
                    _ = s.dataset[i.name + "Call"],
                    p = s.dataset[i.name + "Target"],
                    v;
                p !== void 0 ? v = document.querySelector("".concat(p)) : v = s;
                var T = v.getBoundingClientRect();
                u = T.top + i.instance.scroll.y, h = T.left + i.instance.scroll.x;
                var b = u + v.offsetHeight,
                    E = h + v.offsetWidth;
                f == "false" ? f = !1 : f != null ? f = !0 : f = i.repeat;
                var g = i.getRelativeOffset(d);
                u = u + g[0], b = b - g[1];
                var y = {
                    el: s,
                    targetEl: v,
                    id: c,
                    class: l,
                    top: u,
                    bottom: b,
                    left: h,
                    right: E,
                    offset: d,
                    progress: 0,
                    repeat: f,
                    inView: !1,
                    call: _
                };
                i.els[c] = y, s.classList.contains(l) && i.setInView(i.els[c], c)
            })
        }
    }, {
        key: "updateElements",
        value: function() {
            var i = this;
            Object.entries(this.els).forEach(function(n) {
                var s = dr(n, 2),
                    a = s[0],
                    l = s[1],
                    c = l.targetEl.getBoundingClientRect().top + i.instance.scroll.y,
                    u = c + l.targetEl.offsetHeight,
                    h = i.getRelativeOffset(l.offset);
                i.els[a].top = c + h[0], i.els[a].bottom = u - h[1]
            }), this.hasScrollTicking = !1
        }
    }, {
        key: "getRelativeOffset",
        value: function(i) {
            var n = [0, 0];
            if (i)
                for (var s = 0; s < i.length; s++) typeof i[s] == "string" ? i[s].includes("%") ? n[s] = parseInt(i[s].replace("%", "") * this.windowHeight / 100) : n[s] = parseInt(i[s]) : n[s] = i[s];
            return n
        }
    }, {
        key: "scrollTo",
        value: function(i) {
            var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {},
                s = parseInt(n.offset) || 0,
                a = n.callback ? n.callback : !1;
            if (typeof i == "string") {
                if (i === "top") i = this.html;
                else if (i === "bottom") i = this.html.offsetHeight - window.innerHeight;
                else if (i = document.querySelector(i), !i) return
            } else if (typeof i == "number") i = parseInt(i);
            else if (!(i && i.tagName)) {
                console.warn("`target` parameter is not valid");
                return
            }
            typeof i != "number" ? s = i.getBoundingClientRect().top + s + this.instance.scroll.y : s = i + s;
            var l = function() {
                return parseInt(window.pageYOffset) === parseInt(s)
            };
            if (a)
                if (l()) {
                    a();
                    return
                } else {
                    var c = function u() {
                        l() && (window.removeEventListener("scroll", u), a())
                    };
                    window.addEventListener("scroll", c)
                } window.scrollTo({
                top: s,
                behavior: n.duration === 0 ? "auto" : "smooth"
            })
        }
    }, {
        key: "update",
        value: function() {
            this.addElements(), this.detectElements()
        }
    }, {
        key: "destroy",
        value: function() {
            ci(Ve(r.prototype), "destroy", this).call(this), window.removeEventListener("scroll", this.checkScroll, !1)
        }
    }]), r
}(_a);
*/
/*
object-assign
(c) Sindre Sorhus
@license MIT
*/
var Po = Object.getOwnPropertySymbols,
    lc = Object.prototype.hasOwnProperty,
    cc = Object.prototype.propertyIsEnumerable;

function uc(o) {
    if (o == null) throw new TypeError("Object.assign cannot be called with null or undefined");
    return Object(o)
}

function fc() {
    try {
        if (!Object.assign) return !1;
        var o = new String("abc");
        if (o[5] = "de", Object.getOwnPropertyNames(o)[0] === "5") return !1;
        for (var t = {}, r = 0; r < 10; r++) t["_" + String.fromCharCode(r)] = r;
        var e = Object.getOwnPropertyNames(t).map(function(n) {
            return t[n]
        });
        if (e.join("") !== "0123456789") return !1;
        var i = {};
        return "abcdefghijklmnopqrst".split("").forEach(function(n) {
            i[n] = n
        }), Object.keys(Object.assign({}, i)).join("") === "abcdefghijklmnopqrst"
    } catch {
        return !1
    }
}
var hc = fc() ? Object.assign : function(o, t) {
    for (var r, e = uc(o), i, n = 1; n < arguments.length; n++) {
        r = Object(arguments[n]);
        for (var s in r) lc.call(r, s) && (e[s] = r[s]);
        if (Po) {
            i = Po(r);
            for (var a = 0; a < i.length; a++) cc.call(r, i[a]) && (e[i[a]] = r[i[a]])
        }
    }
    return e
};

function va() {}
va.prototype = {
    on: function(o, t, r) {
        var e = this.e || (this.e = {});
        return (e[o] || (e[o] = [])).push({
            fn: t,
            ctx: r
        }), this
    },
    once: function(o, t, r) {
        var e = this;

        function i() {
            e.off(o, i), t.apply(r, arguments)
        }
        return i._ = t, this.on(o, i, r)
    },
    emit: function(o) {
        var t = [].slice.call(arguments, 1),
            r = ((this.e || (this.e = {}))[o] || []).slice(),
            e = 0,
            i = r.length;
        for (e; e < i; e++) r[e].fn.apply(r[e].ctx, t);
        return this
    },
    off: function(o, t) {
        var r = this.e || (this.e = {}),
            e = r[o],
            i = [];
        if (e && t)
            for (var n = 0, s = e.length; n < s; n++) e[n].fn !== t && e[n].fn._ !== t && i.push(e[n]);
        return i.length ? r[o] = i : delete r[o], this
    }
};
var dc = va,
    pc = ga(function(o, t) {
        (function() {
            var r;
            r = t !== null ? t : this, r.Lethargy = function() {
                function e(i, n, s, a) {
                    this.stability = i != null ? Math.abs(i) : 8, this.sensitivity = n != null ? 1 + Math.abs(n) : 100, this.tolerance = s != null ? 1 + Math.abs(s) : 1.1, this.delay = a ?? 150, this.lastUpDeltas = (function() {
                        var l, c, u;
                        for (u = [], l = 1, c = this.stability * 2; 1 <= c ? l <= c : l >= c; 1 <= c ? l++ : l--) u.push(null);
                        return u
                    }).call(this), this.lastDownDeltas = (function() {
                        var l, c, u;
                        for (u = [], l = 1, c = this.stability * 2; 1 <= c ? l <= c : l >= c; 1 <= c ? l++ : l--) u.push(null);
                        return u
                    }).call(this), this.deltasTimestamp = (function() {
                        var l, c, u;
                        for (u = [], l = 1, c = this.stability * 2; 1 <= c ? l <= c : l >= c; 1 <= c ? l++ : l--) u.push(null);
                        return u
                    }).call(this)
                }
                return e.prototype.check = function(i) {
                    var n;
                    return i = i.originalEvent || i, i.wheelDelta != null ? n = i.wheelDelta : i.deltaY != null ? n = i.deltaY * -40 : (i.detail != null || i.detail === 0) && (n = i.detail * -40), this.deltasTimestamp.push(Date.now()), this.deltasTimestamp.shift(), n > 0 ? (this.lastUpDeltas.push(n), this.lastUpDeltas.shift(), this.isInertia(1)) : (this.lastDownDeltas.push(n), this.lastDownDeltas.shift(), this.isInertia(-1))
                }, e.prototype.isInertia = function(i) {
                    var n, s, a, l, c, u, h;
                    return n = i === -1 ? this.lastDownDeltas : this.lastUpDeltas, n[0] === null ? i : this.deltasTimestamp[this.stability * 2 - 2] + this.delay > Date.now() && n[0] === n[this.stability * 2 - 1] ? !1 : (a = n.slice(0, this.stability), s = n.slice(this.stability, this.stability * 2), h = a.reduce(function(d, f) {
                        return d + f
                    }), c = s.reduce(function(d, f) {
                        return d + f
                    }), u = h / a.length, l = c / s.length, Math.abs(u) < Math.abs(l * this.tolerance) && this.sensitivity < Math.abs(l) ? i : !1)
                }, e.prototype.showLastUpDeltas = function() {
                    return this.lastUpDeltas
                }, e.prototype.showLastDownDeltas = function() {
                    return this.lastDownDeltas
                }, e
            }()
        }).call(oc)
    }),
    ve = function() {
        return {
            hasWheelEvent: "onwheel" in document,
            hasMouseWheelEvent: "onmousewheel" in document,
            hasTouch: "ontouchstart" in window || window.TouchEvent || window.DocumentTouch && document instanceof DocumentTouch,
            hasTouchWin: navigator.msMaxTouchPoints && navigator.msMaxTouchPoints > 1,
            hasPointer: !!window.navigator.msPointerEnabled,
            hasKeyDown: "onkeydown" in document,
            isFirefox: navigator.userAgent.indexOf("Firefox") > -1
        }
    }(),
    _c = Object.prototype.toString,
    gc = Object.prototype.hasOwnProperty,
    mc = function(o) {
        if (!o) return console.warn("bindAll requires at least one argument.");
        var t = Array.prototype.slice.call(arguments, 1);
        if (t.length === 0)
            for (var r in o) gc.call(o, r) && typeof o[r] == "function" && _c.call(o[r]) == "[object Function]" && t.push(r);
        for (var e = 0; e < t.length; e++) {
            var i = t[e];
            o[i] = vc(o[i], o)
        }
    };

function vc(o, t) {
    return function() {
        return o.apply(t, arguments)
    }
}
var yc = pc.Lethargy,
    Xi = "virtualscroll",
    wc = Se,
    Br = {
        LEFT: 37,
        UP: 38,
        RIGHT: 39,
        DOWN: 40,
        SPACE: 32
    };

function Se(o) {
    mc(this, "_onWheel", "_onMouseWheel", "_onTouchStart", "_onTouchMove", "_onKeyDown"), this.el = window, o && o.el && (this.el = o.el, delete o.el), this.options = hc({
        mouseMultiplier: 1,
        touchMultiplier: 2,
        firefoxMultiplier: 15,
        keyStep: 120,
        preventTouch: !1,
        unpreventTouchClass: "vs-touchmove-allowed",
        limitInertia: !1,
        useKeyboard: !0,
        useTouch: !0
    }, o), this.options.limitInertia && (this._lethargy = new yc), this._emitter = new dc, this._event = {
        y: 0,
        x: 0,
        deltaX: 0,
        deltaY: 0
    }, this.touchStartX = null, this.touchStartY = null, this.bodyTouchAction = null, this.options.passive !== void 0 && (this.listenerOptions = {
        passive: this.options.passive
    })
}
Se.prototype._notify = function(o) {
    var t = this._event;
    t.x += t.deltaX, t.y += t.deltaY, this._emitter.emit(Xi, {
        x: t.x,
        y: t.y,
        deltaX: t.deltaX,
        deltaY: t.deltaY,
        originalEvent: o
    })
};
Se.prototype._onWheel = function(o) {
    var t = this.options;
    if (!(this._lethargy && this._lethargy.check(o) === !1)) {
        var r = this._event;
        r.deltaX = o.wheelDeltaX || o.deltaX * -1, r.deltaY = o.wheelDeltaY || o.deltaY * -1, ve.isFirefox && o.deltaMode == 1 && (r.deltaX *= t.firefoxMultiplier, r.deltaY *= t.firefoxMultiplier), r.deltaX *= t.mouseMultiplier, r.deltaY *= t.mouseMultiplier, this._notify(o)
    }
};
Se.prototype._onMouseWheel = function(o) {
    if (!(this.options.limitInertia && this._lethargy.check(o) === !1)) {
        var t = this._event;
        t.deltaX = o.wheelDeltaX ? o.wheelDeltaX : 0, t.deltaY = o.wheelDeltaY ? o.wheelDeltaY : o.wheelDelta, this._notify(o)
    }
};
Se.prototype._onTouchStart = function(o) {
    var t = o.targetTouches ? o.targetTouches[0] : o;
    this.touchStartX = t.pageX, this.touchStartY = t.pageY
};
Se.prototype._onTouchMove = function(o) {
    var t = this.options;
    t.preventTouch && !o.target.classList.contains(t.unpreventTouchClass) && o.preventDefault();
    var r = this._event,
        e = o.targetTouches ? o.targetTouches[0] : o;
    r.deltaX = (e.pageX - this.touchStartX) * t.touchMultiplier, r.deltaY = (e.pageY - this.touchStartY) * t.touchMultiplier, this.touchStartX = e.pageX, this.touchStartY = e.pageY, this._notify(o)
};
Se.prototype._onKeyDown = function(o) {
    var t = this._event;
    t.deltaX = t.deltaY = 0;
    var r = window.innerHeight - 40;
    switch (o.keyCode) {
        case Br.LEFT:
        case Br.UP:
            t.deltaY = this.options.keyStep;
            break;
        case Br.RIGHT:
        case Br.DOWN:
            t.deltaY = -this.options.keyStep;
            break;
        case o.shiftKey:
            t.deltaY = r;
            break;
        case Br.SPACE:
            t.deltaY = -r;
            break;
        default:
            return
    }
    this._notify(o)
};
Se.prototype._bind = function() {
    ve.hasWheelEvent && this.el.addEventListener("wheel", this._onWheel, this.listenerOptions), ve.hasMouseWheelEvent && this.el.addEventListener("mousewheel", this._onMouseWheel, this.listenerOptions), ve.hasTouch && this.options.useTouch && (this.el.addEventListener("touchstart", this._onTouchStart, this.listenerOptions), this.el.addEventListener("touchmove", this._onTouchMove, this.listenerOptions)), ve.hasPointer && ve.hasTouchWin && (this.bodyTouchAction = document.body.style.msTouchAction, document.body.style.msTouchAction = "none", this.el.addEventListener("MSPointerDown", this._onTouchStart, !0), this.el.addEventListener("MSPointerMove", this._onTouchMove, !0)), ve.hasKeyDown && this.options.useKeyboard && document.addEventListener("keydown", this._onKeyDown)
};
Se.prototype._unbind = function() {
    ve.hasWheelEvent && this.el.removeEventListener("wheel", this._onWheel), ve.hasMouseWheelEvent && this.el.removeEventListener("mousewheel", this._onMouseWheel), ve.hasTouch && (this.el.removeEventListener("touchstart", this._onTouchStart), this.el.removeEventListener("touchmove", this._onTouchMove)), ve.hasPointer && ve.hasTouchWin && (document.body.style.msTouchAction = this.bodyTouchAction, this.el.removeEventListener("MSPointerDown", this._onTouchStart, !0), this.el.removeEventListener("MSPointerMove", this._onTouchMove, !0)), ve.hasKeyDown && this.options.useKeyboard && document.removeEventListener("keydown", this._onKeyDown)
};
Se.prototype.on = function(o, t) {
    this._emitter.on(Xi, o, t);
    var r = this._emitter.e;
    r && r[Xi] && r[Xi].length === 1 && this._bind()
};
Se.prototype.off = function(o, t) {
    this._emitter.off(Xi, o, t);
    var r = this._emitter.e;
    (!r[Xi] || r[Xi].length <= 0) && this._unbind()
};
Se.prototype.reset = function() {
    var o = this._event;
    o.x = 0, o.y = 0
};
Se.prototype.destroy = function() {
    this._emitter.off(), this._unbind()
};

function us(o, t, r) {
    return (1 - r) * o + r * t
}

function Ee(o) {
    var t = {};
    if (window.getComputedStyle) {
        var r = getComputedStyle(o),
            e = r.transform || r.webkitTransform || r.mozTransform,
            i = e.match(/^matrix3d\((.+)\)$/);
        return i ? (t.x = i ? parseFloat(i[1].split(", ")[12]) : 0, t.y = i ? parseFloat(i[1].split(", ")[13]) : 0) : (i = e.match(/^matrix\((.+)\)$/), t.x = i ? parseFloat(i[1].split(", ")[4]) : 0, t.y = i ? parseFloat(i[1].split(", ")[5]) : 0), t
    }
}

function fs(o) {
    for (var t = []; o && o !== document; o = o.parentNode) t.push(o);
    return t
}
var bc = 4,
    xc = .001,
    Tc = 1e-7,
    Sc = 10,
    Nr = 11,
    wn = 1 / (Nr - 1),
    kc = typeof Float32Array == "function";

function ya(o, t) {
    return 1 - 3 * t + 3 * o
}

function wa(o, t) {
    return 3 * t - 6 * o
}

function ba(o) {
    return 3 * o
}

function Vn(o, t, r) {
    return ((ya(t, r) * o + wa(t, r)) * o + ba(t)) * o
}

function xa(o, t, r) {
    return 3 * ya(t, r) * o * o + 2 * wa(t, r) * o + ba(t)
}

function Ec(o, t, r, e, i) {
    var n, s, a = 0;
    do s = t + (r - t) / 2, n = Vn(s, e, i) - o, n > 0 ? r = s : t = s; while (Math.abs(n) > Tc && ++a < Sc);
    return s
}

function Oc(o, t, r, e) {
    for (var i = 0; i < bc; ++i) {
        var n = xa(t, r, e);
        if (n === 0) return t;
        var s = Vn(t, r, e) - o;
        t -= s / n
    }
    return t
}

function Cc(o) {
    return o
}
var Pc = function(t, r, e, i) {
        if (!(0 <= t && t <= 1 && 0 <= e && e <= 1)) throw new Error("bezier x values must be in [0, 1] range");
        if (t === r && e === i) return Cc;
        for (var n = kc ? new Float32Array(Nr) : new Array(Nr), s = 0; s < Nr; ++s) n[s] = Vn(s * wn, t, e);

        function a(l) {
            for (var c = 0, u = 1, h = Nr - 1; u !== h && n[u] <= l; ++u) c += wn;
            --u;
            var d = (l - n[u]) / (n[u + 1] - n[u]),
                f = c + d * wn,
                _ = xa(f, t, e);
            return _ >= xc ? Oc(l, f, t, e) : _ === 0 ? f : Ec(l, c, c + wn, t, e)
        }
        return function(c) {
            return c === 0 ? 0 : c === 1 ? 1 : Vn(a(c), r, i)
        }
    },
    ni = {
        LEFT: 37,
        UP: 38,
        RIGHT: 39,
        DOWN: 40,
        SPACE: 32,
        TAB: 9,
        PAGEUP: 33,
        PAGEDOWN: 34,
        HOME: 36,
        END: 35
    },
    Ac = function(o) {
        fa(r, o);
        var t = da(r);

        function r() {
            var e, i = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
            return is(this, r), history.scrollRestoration && (history.scrollRestoration = "manual"), window.scrollTo(0, 0), e = t.call(this, i), e.inertia && (e.lerp = e.inertia * .1), e.isScrolling = !1, e.isDraggingScrollbar = !1, e.isTicking = !1, e.hasScrollTicking = !1, e.parallaxElements = {}, e.stop = !1, e.scrollbarContainer = i.scrollbarContainer, e.checkKey = e.checkKey.bind(ha(e)), window.addEventListener("keydown", e.checkKey, !1), e
        }
        return rs(r, [{
            key: "init",
            value: function() {
                var i = this;
                this.html.classList.add(this.smoothClass), this.html.setAttribute("data-".concat(this.name, "-direction"), this.direction), this.instance = cs({
                    delta: {
                        x: this.initPosition.x,
                        y: this.initPosition.y
                    },
                    scroll: {
                        x: this.initPosition.x,
                        y: this.initPosition.y
                    }
                }, this.instance), this.vs = new wc({
                    el: this.scrollFromAnywhere ? document : this.el,
                    mouseMultiplier: navigator.platform.indexOf("Win") > -1 ? 1 : .4,
                    firefoxMultiplier: this.firefoxMultiplier,
                    touchMultiplier: this.touchMultiplier,
                    useKeyboard: !1,
                    passive: !0
                }), this.vs.on(function(n) {
                    i.stop || i.isDraggingScrollbar || requestAnimationFrame(function() {
                        i.updateDelta(n), i.isScrolling || i.startScrolling()
                    })
                }), this.setScrollLimit(), this.initScrollBar(), this.addSections(), this.addElements(), this.checkScroll(!0), this.transformElements(!0, !0), ci(Ve(r.prototype), "init", this).call(this)
            }
        }, {
            key: "setScrollLimit",
            value: function() {
                if (this.instance.limit.y = this.el.offsetHeight - this.windowHeight, this.direction === "horizontal") {
                    for (var i = 0, n = this.el.children, s = 0; s < n.length; s++) i += n[s].offsetWidth;
                    this.instance.limit.x = i - this.windowWidth
                }
            }
        }, {
            key: "startScrolling",
            value: function() {
                this.startScrollTs = Date.now(), this.isScrolling = !0, this.checkScroll(), this.html.classList.add(this.scrollingClass)
            }
        }, {
            key: "stopScrolling",
            value: function() {
                cancelAnimationFrame(this.checkScrollRaf), this.startScrollTs = void 0, this.scrollToRaf && (cancelAnimationFrame(this.scrollToRaf), this.scrollToRaf = null), this.isScrolling = !1, this.instance.scroll.y = Math.round(this.instance.scroll.y), this.html.classList.remove(this.scrollingClass)
            }
        }, {
            key: "checkKey",
            value: function(i) {
                var n = this;
                if (this.stop) {
                    i.keyCode == ni.TAB && requestAnimationFrame(function() {
                        n.html.scrollTop = 0, document.body.scrollTop = 0, n.html.scrollLeft = 0, document.body.scrollLeft = 0
                    });
                    return
                }
                switch (i.keyCode) {
                    case ni.TAB:
                        requestAnimationFrame(function() {
                            n.html.scrollTop = 0, document.body.scrollTop = 0, n.html.scrollLeft = 0, document.body.scrollLeft = 0, n.scrollTo(document.activeElement, {
                                offset: -window.innerHeight / 2
                            })
                        });
                        break;
                    case ni.UP:
                        this.isActiveElementScrollSensitive() && (this.instance.delta[this.directionAxis] -= 240);
                        break;
                    case ni.DOWN:
                        this.isActiveElementScrollSensitive() && (this.instance.delta[this.directionAxis] += 240);
                        break;
                    case ni.PAGEUP:
                        this.instance.delta[this.directionAxis] -= window.innerHeight;
                        break;
                    case ni.PAGEDOWN:
                        this.instance.delta[this.directionAxis] += window.innerHeight;
                        break;
                    case ni.HOME:
                        this.instance.delta[this.directionAxis] -= this.instance.limit[this.directionAxis];
                        break;
                    case ni.END:
                        this.instance.delta[this.directionAxis] += this.instance.limit[this.directionAxis];
                        break;
                    case ni.SPACE:
                        this.isActiveElementScrollSensitive() && (i.shiftKey ? this.instance.delta[this.directionAxis] -= window.innerHeight : this.instance.delta[this.directionAxis] += window.innerHeight);
                        break;
                    default:
                        return
                }
                this.instance.delta[this.directionAxis] < 0 && (this.instance.delta[this.directionAxis] = 0), this.instance.delta[this.directionAxis] > this.instance.limit[this.directionAxis] && (this.instance.delta[this.directionAxis] = this.instance.limit[this.directionAxis]), this.stopScrolling(), this.isScrolling = !0, this.checkScroll(), this.html.classList.add(this.scrollingClass)
            }
        }, {
            key: "isActiveElementScrollSensitive",
            value: function() {
                return !(document.activeElement instanceof HTMLInputElement) && !(document.activeElement instanceof HTMLTextAreaElement) && !(document.activeElement instanceof HTMLButtonElement) && !(document.activeElement instanceof HTMLSelectElement)
            }
        }, {
            key: "checkScroll",
            value: function() {
                var i = this,
                    n = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : !1;
                if (n || this.isScrolling || this.isDraggingScrollbar) {
                    this.hasScrollTicking || (this.checkScrollRaf = requestAnimationFrame(function() {
                        return i.checkScroll()
                    }), this.hasScrollTicking = !0), this.updateScroll();
                    var s = Math.abs(this.instance.delta[this.directionAxis] - this.instance.scroll[this.directionAxis]),
                        a = Date.now() - this.startScrollTs;
                    if (!this.animatingScroll && a > 100 && (s < .5 && this.instance.delta[this.directionAxis] != 0 || s < .5 && this.instance.delta[this.directionAxis] == 0) && this.stopScrolling(), Object.entries(this.sections).forEach(function(c) {
                            var u = dr(c, 2);
                            u[0];
                            var h = u[1];
                            h.persistent || i.instance.scroll[i.directionAxis] > h.offset[i.directionAxis] && i.instance.scroll[i.directionAxis] < h.limit[i.directionAxis] ? (i.direction === "horizontal" ? i.transform(h.el, -i.instance.scroll[i.directionAxis], 0) : i.transform(h.el, 0, -i.instance.scroll[i.directionAxis]), h.inView || (h.inView = !0, h.el.style.opacity = 1, h.el.style.pointerEvents = "all", h.el.setAttribute("data-".concat(i.name, "-section-inview"), ""))) : ((h.inView || n) && (h.inView = !1, h.el.style.opacity = 0, h.el.style.pointerEvents = "none", h.el.removeAttribute("data-".concat(i.name, "-section-inview"))), i.transform(h.el, 0, 0))
                        }), this.getDirection && this.addDirection(), this.getSpeed && (this.addSpeed(), this.speedTs = Date.now()), this.detectElements(), this.transformElements(), this.hasScrollbar) {
                        var l = this.instance.scroll[this.directionAxis] / this.instance.limit[this.directionAxis] * this.scrollBarLimit[this.directionAxis];
                        this.direction === "horizontal" ? this.transform(this.scrollbarThumb, l, 0) : this.transform(this.scrollbarThumb, 0, l)
                    }
                    ci(Ve(r.prototype), "checkScroll", this).call(this), this.hasScrollTicking = !1
                }
            }
        }, {
            key: "resize",
            value: function() {
                this.windowHeight = window.innerHeight, this.windowWidth = window.innerWidth, this.checkContext(), this.windowMiddle = {
                    x: this.windowWidth / 2,
                    y: this.windowHeight / 2
                }, this.update()
            }
        }, {
            key: "updateDelta",
            value: function(i) {
                var n, s = this[this.context] && this[this.context].gestureDirection ? this[this.context].gestureDirection : this.gestureDirection;
                s === "both" ? n = i.deltaX + i.deltaY : s === "vertical" ? n = i.deltaY : s === "horizontal" ? n = i.deltaX : n = i.deltaY, this.instance.delta[this.directionAxis] -= n * this.multiplier, this.instance.delta[this.directionAxis] < 0 && (this.instance.delta[this.directionAxis] = 0), this.instance.delta[this.directionAxis] > this.instance.limit[this.directionAxis] && (this.instance.delta[this.directionAxis] = this.instance.limit[this.directionAxis])
            }
        }, {
            key: "updateScroll",
            value: function(i) {
                this.isScrolling || this.isDraggingScrollbar ? this.instance.scroll[this.directionAxis] = us(this.instance.scroll[this.directionAxis], this.instance.delta[this.directionAxis], this.lerp) : this.instance.scroll[this.directionAxis] > this.instance.limit[this.directionAxis] ? this.setScroll(this.instance.scroll[this.directionAxis], this.instance.limit[this.directionAxis]) : this.instance.scroll.y < 0 ? this.setScroll(this.instance.scroll[this.directionAxis], 0) : this.setScroll(this.instance.scroll[this.directionAxis], this.instance.delta[this.directionAxis])
            }
        }, {
            key: "addDirection",
            value: function() {
                this.instance.delta.y > this.instance.scroll.y ? this.instance.direction !== "down" && (this.instance.direction = "down") : this.instance.delta.y < this.instance.scroll.y && this.instance.direction !== "up" && (this.instance.direction = "up"), this.instance.delta.x > this.instance.scroll.x ? this.instance.direction !== "right" && (this.instance.direction = "right") : this.instance.delta.x < this.instance.scroll.x && this.instance.direction !== "left" && (this.instance.direction = "left")
            }
        }, {
            key: "addSpeed",
            value: function() {
                this.instance.delta[this.directionAxis] != this.instance.scroll[this.directionAxis] ? this.instance.speed = (this.instance.delta[this.directionAxis] - this.instance.scroll[this.directionAxis]) / Math.max(1, Date.now() - this.speedTs) : this.instance.speed = 0
            }
        }, {
            key: "initScrollBar",
            value: function() {
                if (this.scrollbar = document.createElement("span"), this.scrollbarThumb = document.createElement("span"), this.scrollbar.classList.add("".concat(this.scrollbarClass)), this.scrollbarThumb.classList.add("".concat(this.scrollbarClass, "_thumb")), this.scrollbar.append(this.scrollbarThumb), this.scrollbarContainer ? this.scrollbarContainer.append(this.scrollbar) : document.body.append(this.scrollbar), this.getScrollBar = this.getScrollBar.bind(this), this.releaseScrollBar = this.releaseScrollBar.bind(this), this.moveScrollBar = this.moveScrollBar.bind(this), this.scrollbarThumb.addEventListener("mousedown", this.getScrollBar), window.addEventListener("mouseup", this.releaseScrollBar), window.addEventListener("mousemove", this.moveScrollBar), this.hasScrollbar = !1, this.direction == "horizontal") {
                    if (this.instance.limit.x + this.windowWidth <= this.windowWidth) return
                } else if (this.instance.limit.y + this.windowHeight <= this.windowHeight) return;
                this.hasScrollbar = !0, this.scrollbarBCR = this.scrollbar.getBoundingClientRect(), this.scrollbarHeight = this.scrollbarBCR.height, this.scrollbarWidth = this.scrollbarBCR.width, this.direction === "horizontal" ? this.scrollbarThumb.style.width = "".concat(this.scrollbarWidth * this.scrollbarWidth / (this.instance.limit.x + this.scrollbarWidth), "px") : this.scrollbarThumb.style.height = "".concat(this.scrollbarHeight * this.scrollbarHeight / (this.instance.limit.y + this.scrollbarHeight), "px"), this.scrollbarThumbBCR = this.scrollbarThumb.getBoundingClientRect(), this.scrollBarLimit = {
                    x: this.scrollbarWidth - this.scrollbarThumbBCR.width,
                    y: this.scrollbarHeight - this.scrollbarThumbBCR.height
                }
            }
        }, {
            key: "reinitScrollBar",
            value: function() {
                if (this.hasScrollbar = !1, this.direction == "horizontal") {
                    if (this.instance.limit.x + this.windowWidth <= this.windowWidth) return
                } else if (this.instance.limit.y + this.windowHeight <= this.windowHeight) return;
                this.hasScrollbar = !0, this.scrollbarBCR = this.scrollbar.getBoundingClientRect(), this.scrollbarHeight = this.scrollbarBCR.height, this.scrollbarWidth = this.scrollbarBCR.width, this.direction === "horizontal" ? this.scrollbarThumb.style.width = "".concat(this.scrollbarWidth * this.scrollbarWidth / (this.instance.limit.x + this.scrollbarWidth), "px") : this.scrollbarThumb.style.height = "".concat(this.scrollbarHeight * this.scrollbarHeight / (this.instance.limit.y + this.scrollbarHeight), "px"), this.scrollbarThumbBCR = this.scrollbarThumb.getBoundingClientRect(), this.scrollBarLimit = {
                    x: this.scrollbarWidth - this.scrollbarThumbBCR.width,
                    y: this.scrollbarHeight - this.scrollbarThumbBCR.height
                }
            }
        }, {
            key: "destroyScrollBar",
            value: function() {
                this.scrollbarThumb.removeEventListener("mousedown", this.getScrollBar), window.removeEventListener("mouseup", this.releaseScrollBar), window.removeEventListener("mousemove", this.moveScrollBar), this.scrollbar.remove()
            }
        }, {
            key: "getScrollBar",
            value: function(i) {
                this.isDraggingScrollbar = !0, this.checkScroll(), this.html.classList.remove(this.scrollingClass), this.html.classList.add(this.draggingClass)
            }
        }, {
            key: "releaseScrollBar",
            value: function(i) {
                this.isDraggingScrollbar = !1, this.isScrolling && this.html.classList.add(this.scrollingClass), this.html.classList.remove(this.draggingClass)
            }
        }, {
            key: "moveScrollBar",
            value: function(i) {
                var n = this;
                this.isDraggingScrollbar && requestAnimationFrame(function() {
                    var s = (i.clientX - n.scrollbarBCR.left) * 100 / n.scrollbarWidth * n.instance.limit.x / 100,
                        a = (i.clientY - n.scrollbarBCR.top) * 100 / n.scrollbarHeight * n.instance.limit.y / 100;
                    a > 0 && a < n.instance.limit.y && (n.instance.delta.y = a), s > 0 && s < n.instance.limit.x && (n.instance.delta.x = s)
                })
            }
        }, {
            key: "addElements",
            value: function() {
                var i = this;
                this.els = {}, this.parallaxElements = {};
                var n = this.el.querySelectorAll("[data-".concat(this.name, "]"));
                n.forEach(function(s, a) {
                    var l = fs(s),
                        c = Object.entries(i.sections).map(function(J) {
                            var m = dr(J, 2);
                            m[0];
                            var tt = m[1];
                            return tt
                        }).find(function(J) {
                            return l.includes(J.el)
                        }),
                        u = s.dataset[i.name + "Class"] || i.class,
                        h = typeof s.dataset[i.name + "Id"] == "string" ? s.dataset[i.name + "Id"] : "el" + a,
                        d, f, _ = s.dataset[i.name + "Repeat"],
                        p = s.dataset[i.name + "Call"],
                        v = s.dataset[i.name + "Position"],
                        T = s.dataset[i.name + "Delay"],
                        b = s.dataset[i.name + "Direction"],
                        E = typeof s.dataset[i.name + "Sticky"] == "string",
                        g = s.dataset[i.name + "Speed"] ? parseFloat(s.dataset[i.name + "Speed"]) / 10 : !1,
                        y = typeof s.dataset[i.name + "Offset"] == "string" ? s.dataset[i.name + "Offset"].split(",") : i.offset,
                        S = s.dataset[i.name + "Target"],
                        w;
                    S !== void 0 ? w = document.querySelector("".concat(S)) : w = s;
                    var C = w.getBoundingClientRect();
                    c === null || c.inView ? (d = C.top + i.instance.scroll.y - Ee(w).y, f = C.left + i.instance.scroll.x - Ee(w).x) : (d = C.top - Ee(c.el).y - Ee(w).y, f = C.left - Ee(c.el).x - Ee(w).x);
                    var k = d + w.offsetHeight,
                        O = f + w.offsetWidth,
                        L = {
                            x: (O - f) / 2 + f,
                            y: (k - d) / 2 + d
                        };
                    if (E) {
                        var P = s.getBoundingClientRect(),
                            W = P.top,
                            N = P.left,
                            X = {
                                x: N - f,
                                y: W - d
                            };
                        d += window.innerHeight, f += window.innerWidth, k = W + w.offsetHeight - s.offsetHeight - X[i.directionAxis], O = N + w.offsetWidth - s.offsetWidth - X[i.directionAxis], L = {
                            x: (O - f) / 2 + f,
                            y: (k - d) / 2 + d
                        }
                    }
                    _ == "false" ? _ = !1 : _ != null ? _ = !0 : _ = i.repeat;
                    var B = [0, 0];
                    if (y)
                        if (i.direction === "horizontal") {
                            for (var M = 0; M < y.length; M++) typeof y[M] == "string" ? y[M].includes("%") ? B[M] = parseInt(y[M].replace("%", "") * i.windowWidth / 100) : B[M] = parseInt(y[M]) : B[M] = y[M];
                            f = f + B[0], O = O - B[1]
                        } else {
                            for (var M = 0; M < y.length; M++) typeof y[M] == "string" ? y[M].includes("%") ? B[M] = parseInt(y[M].replace("%", "") * i.windowHeight / 100) : B[M] = parseInt(y[M]) : B[M] = y[M];
                            d = d + B[0], k = k - B[1]
                        } var $ = {
                        el: s,
                        id: h,
                        class: u,
                        section: c,
                        top: d,
                        middle: L,
                        bottom: k,
                        left: f,
                        right: O,
                        offset: y,
                        progress: 0,
                        repeat: _,
                        inView: !1,
                        call: p,
                        speed: g,
                        delay: T,
                        position: v,
                        target: w,
                        direction: b,
                        sticky: E
                    };
                    i.els[h] = $, s.classList.contains(u) && i.setInView(i.els[h], h), (g !== !1 || E) && (i.parallaxElements[h] = $)
                })
            }
        }, {
            key: "addSections",
            value: function() {
                var i = this;
                this.sections = {};
                var n = this.el.querySelectorAll("[data-".concat(this.name, "-section]"));
                n.length === 0 && (n = [this.el]), n.forEach(function(s, a) {
                    var l = typeof s.dataset[i.name + "Id"] == "string" ? s.dataset[i.name + "Id"] : "section" + a,
                        c = s.getBoundingClientRect(),
                        u = {
                            x: c.left - window.innerWidth * 1.5 - Ee(s).x,
                            y: c.top - window.innerHeight * 1.5 - Ee(s).y
                        },
                        h = {
                            x: u.x + c.width + window.innerWidth * 2,
                            y: u.y + c.height + window.innerHeight * 2
                        },
                        d = typeof s.dataset[i.name + "Persistent"] == "string";
                    s.setAttribute("data-scroll-section-id", l);
                    var f = {
                        el: s,
                        offset: u,
                        limit: h,
                        inView: !1,
                        persistent: d,
                        id: l
                    };
                    i.sections[l] = f
                })
            }
        }, {
            key: "transform",
            value: function(i, n, s, a) {
                var l;
                if (!a) l = "matrix3d(1,0,0.00,0,0.00,1,0.00,0,0,0,1,0,".concat(n, ",").concat(s, ",0,1)");
                else {
                    var c = Ee(i),
                        u = us(c.x, n, a),
                        h = us(c.y, s, a);
                    l = "matrix3d(1,0,0.00,0,0.00,1,0.00,0,0,0,1,0,".concat(u, ",").concat(h, ",0,1)")
                }
                i.style.webkitTransform = l, i.style.msTransform = l, i.style.transform = l
            }
        }, {
            key: "transformElements",
            value: function(i) {
                var n = this,
                    s = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : !1,
                    a = this.instance.scroll.x + this.windowWidth,
                    l = this.instance.scroll.y + this.windowHeight,
                    c = {
                        x: this.instance.scroll.x + this.windowMiddle.x,
                        y: this.instance.scroll.y + this.windowMiddle.y
                    };
                Object.entries(this.parallaxElements).forEach(function(u) {
                    var h = dr(u, 2);
                    h[0];
                    var d = h[1],
                        f = !1;
                    if (i && (f = 0), d.inView || s) switch (d.position) {
                        case "top":
                            f = n.instance.scroll[n.directionAxis] * -d.speed;
                            break;
                        case "elementTop":
                            f = (l - d.top) * -d.speed;
                            break;
                        case "bottom":
                            f = (n.instance.limit[n.directionAxis] - l + n.windowHeight) * d.speed;
                            break;
                        case "left":
                            f = n.instance.scroll[n.directionAxis] * -d.speed;
                            break;
                        case "elementLeft":
                            f = (a - d.left) * -d.speed;
                            break;
                        case "right":
                            f = (n.instance.limit[n.directionAxis] - a + n.windowHeight) * d.speed;
                            break;
                        default:
                            f = (c[n.directionAxis] - d.middle[n.directionAxis]) * -d.speed;
                            break
                    }
                    d.sticky && (d.inView ? n.direction === "horizontal" ? f = n.instance.scroll.x - d.left + window.innerWidth : f = n.instance.scroll.y - d.top + window.innerHeight : n.direction === "horizontal" ? n.instance.scroll.x < d.left - window.innerWidth && n.instance.scroll.x < d.left - window.innerWidth / 2 ? f = 0 : n.instance.scroll.x > d.right && n.instance.scroll.x > d.right + 100 ? f = d.right - d.left + window.innerWidth : f = !1 : n.instance.scroll.y < d.top - window.innerHeight && n.instance.scroll.y < d.top - window.innerHeight / 2 ? f = 0 : n.instance.scroll.y > d.bottom && n.instance.scroll.y > d.bottom + 100 ? f = d.bottom - d.top + window.innerHeight : f = !1), f !== !1 && (d.direction === "horizontal" || n.direction === "horizontal" && d.direction !== "vertical" ? n.transform(d.el, f, 0, i ? !1 : d.delay) : n.transform(d.el, 0, f, i ? !1 : d.delay))
                })
            }
        }, {
            key: "scrollTo",
            value: function(i) {
                var n = this,
                    s = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {},
                    a = parseInt(s.offset) || 0,
                    l = isNaN(parseInt(s.duration)) ? 1e3 : parseInt(s.duration),
                    c = s.easing || [.25, 0, .35, 1],
                    u = !!s.disableLerp,
                    h = s.callback ? s.callback : !1;
                if (c = Pc.apply(void 0, Jl(c)), typeof i == "string") {
                    if (i === "top") i = 0;
                    else if (i === "bottom") i = this.instance.limit.y;
                    else if (i === "left") i = 0;
                    else if (i === "right") i = this.instance.limit.x;
                    else if (i = document.querySelector(i), !i) return
                } else if (typeof i == "number") i = parseInt(i);
                else if (!(i && i.tagName)) {
                    console.warn("`target` parameter is not valid");
                    return
                }
                if (typeof i != "number") {
                    var d = fs(i).includes(this.el);
                    if (!d) return;
                    var f = i.getBoundingClientRect(),
                        _ = f.top,
                        p = f.left,
                        v = fs(i),
                        T = v.find(function(k) {
                            return Object.entries(n.sections).map(function(O) {
                                var L = dr(O, 2);
                                L[0];
                                var P = L[1];
                                return P
                            }).find(function(O) {
                                return O.el == k
                            })
                        }),
                        b = 0;
                    T ? b = Ee(T)[this.directionAxis] : b = -this.instance.scroll[this.directionAxis], this.direction === "horizontal" ? a = p + a - b : a = _ + a - b
                } else a = i + a;
                var E = parseFloat(this.instance.delta[this.directionAxis]),
                    g = Math.max(0, Math.min(a, this.instance.limit[this.directionAxis])),
                    y = g - E,
                    S = function(O) {
                        u ? n.direction === "horizontal" ? n.setScroll(E + y * O, n.instance.delta.y) : n.setScroll(n.instance.delta.x, E + y * O) : n.instance.delta[n.directionAxis] = E + y * O
                    };
                this.animatingScroll = !0, this.stopScrolling(), this.startScrolling();
                var w = Date.now(),
                    C = function k() {
                        var O = (Date.now() - w) / l;
                        O > 1 ? (S(1), n.animatingScroll = !1, l == 0 && n.update(), h && h()) : (n.scrollToRaf = requestAnimationFrame(k), S(c(O)))
                    };
                C()
            }
        }, {
            key: "update",
            value: function() {
                this.setScrollLimit(), this.addSections(), this.addElements(), this.detectElements(), this.updateScroll(), this.transformElements(!0), this.reinitScrollBar(), this.checkScroll(!0)
            }
        }, {
            key: "startScroll",
            value: function() {
                this.stop = !1
            }
        }, {
            key: "stopScroll",
            value: function() {
                this.stop = !0
            }
        }, {
            key: "setScroll",
            value: function(i, n) {
                this.instance = cs(cs({}, this.instance), {}, {
                    scroll: {
                        x: i,
                        y: n
                    },
                    delta: {
                        x: i,
                        y: n
                    },
                    speed: 0
                })
            }
        }, {
            key: "destroy",
            value: function() {
                ci(Ve(r.prototype), "destroy", this).call(this), this.stopScrolling(), this.html.classList.remove(this.smoothClass), this.vs.destroy(), this.destroyScrollBar(), window.removeEventListener("keydown", this.checkKey, !1)
            }
        }]), r
    }(_a),
    Mc = function() {
        function o() {
            var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
            is(this, o), this.options = t, Object.assign(this, mr, t), this.smartphone = mr.smartphone, t.smartphone && Object.assign(this.smartphone, t.smartphone), this.tablet = mr.tablet, t.tablet && Object.assign(this.tablet, t.tablet), !this.smooth && this.direction == "horizontal" && console.warn("🚨 `smooth:false` & `horizontal` direction are not yet compatible"), !this.tablet.smooth && this.tablet.direction == "horizontal" && console.warn("🚨 `smooth:false` & `horizontal` direction are not yet compatible (tablet)"), !this.smartphone.smooth && this.smartphone.direction == "horizontal" && console.warn("🚨 `smooth:false` & `horizontal` direction are not yet compatible (smartphone)"), this.init()
        }
        return rs(o, [{
            key: "init",
            value: function() {
                if (this.options.isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1 || window.innerWidth < this.tablet.breakpoint, this.options.isTablet = this.options.isMobile && window.innerWidth >= this.tablet.breakpoint, this.smooth && !this.options.isMobile || this.tablet.smooth && this.options.isTablet || this.smartphone.smooth && this.options.isMobile && !this.options.isTablet ? this.scroll = new Ac(this.options) : this.scroll = new ac(this.options), this.scroll.init(), window.location.hash) {
                    var r = window.location.hash.slice(1, window.location.hash.length),
                        e = document.getElementById(r);
                    e && this.scroll.scrollTo(e)
                }
            }
        }, {
            key: "update",
            value: function() {
                this.scroll.update()
            }
        }, {
            key: "start",
            value: function() {
                this.scroll.startScroll()
            }
        }, {
            key: "stop",
            value: function() {
                this.scroll.stopScroll()
            }
        }, {
            key: "scrollTo",
            value: function(r, e) {
                this.scroll.scrollTo(r, e)
            }
        }, {
            key: "setScroll",
            value: function(r, e) {
                this.scroll.setScroll(r, e)
            }
        }, {
            key: "on",
            value: function(r, e) {
                this.scroll.setEvents(r, e)
            }
        }, {
            key: "off",
            value: function(r, e) {
                this.scroll.unsetEvents(r, e)
            }
        }, {
            key: "destroy",
            value: function() {
                this.scroll.destroy()
            }
        }]), o
    }();

function si(o) {
    if (o === void 0) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return o
}

function Ta(o, t) {
    o.prototype = Object.create(t.prototype), o.prototype.constructor = o, o.__proto__ = t
}

var xe = {
        autoSleep: 120,
        force3D: "auto",
        nullTargetWarn: 1,
        units: {
            lineHeight: ""
        }
    },
    kr = {
        duration: .5,
        overwrite: !1,
        delay: 0
    },
    io, Xt, ot, Ae = 1e8,
    rt = 1 / Ae,
    As = Math.PI * 2,
    Dc = As / 4,
    Rc = 0,
    Sa = Math.sqrt,
    Lc = Math.cos,
    Ic = Math.sin,
    Dt = function(t) {
        return typeof t == "string"
    },
    dt = function(t) {
        return typeof t == "function"
    },
    fi = function(t) {
        return typeof t == "number"
    },
    ro = function(t) {
        return typeof t > "u"
    },
    ti = function(t) {
        return typeof t == "object"
    },
    oe = function(t) {
        return t !== !1
    },
    no = function() {
        return typeof window < "u"
    },
    bn = function(t) {
        return dt(t) || Dt(t)
    },
    ka = typeof ArrayBuffer == "function" && ArrayBuffer.isView || function() {},
    Ut = Array.isArray,
    Ms = /(?:-?\.?\d|\.)+/gi,
    Ea = /[-+=.]*\d+[.e\-+]*\d*[e\-+]*\d*/g,
    pr = /[-+=.]*\d+[.e-]*\d*[a-z%]*/g,
    hs = /[-+=.]*\d+\.?\d*(?:e-|e\+)?\d*/gi,
    Oa = /[+-]=-?[.\d]+/,
    Ca = /[^,'"\[\]\s]+/gi,
    Bc = /^[+\-=e\s\d]*\d+[.\d]*([a-z]*|%)\s*$/i,
    ct, $e, Ds, so, Te = {},
    Xn = {},
    Pa, Aa = function(t) {
        return (Xn = Qi(t, Te)) && ue
    },
    oo = function(t, r) {
        return console.warn("Invalid property", t, "set to", r, "Missing plugin? gsap.registerPlugin()")
    },
    an = function(t, r) {
        return !r && console.warn(t)
    },
    Ma = function(t, r) {
        return t && (Te[t] = r) && Xn && (Xn[t] = r) || Te
    },
    ln = function() {
        return 0
    },
    zc = {
        suppressEvents: !0,
        isStart: !0,
        kill: !1
    },
    Ln = {
        suppressEvents: !0,
        kill: !1
    },
    Fc = {
        suppressEvents: !0
    },
    ao = {},
    ki = [],
    Rs = {},
    Da, me = {},
    ds = {},
    Ao = 30,
    In = [],
    lo = "",
    co = function(t) {
        var r = t[0],
            e, i;
        if (ti(r) || dt(r) || (t = [t]), !(e = (r._gsap || {}).harness)) {
            for (i = In.length; i-- && !In[i].targetTest(r););
            e = In[i]
        }
        for (i = t.length; i--;) t[i] && (t[i]._gsap || (t[i]._gsap = new il(t[i], e))) || t.splice(i, 1);
        return t
    },
    Ui = function(t) {
        return t._gsap || co(Me(t))[0]._gsap
    },
    Ra = function(t, r, e) {
        return (e = t[r]) && dt(e) ? t[r]() : ro(e) && t.getAttribute && t.getAttribute(r) || e
    },
    ae = function(t, r) {
        return (t = t.split(",")).forEach(r) || t
    },
    mt = function(t) {
        return Math.round(t * 1e5) / 1e5 || 0
    },
    Mt = function(t) {
        return Math.round(t * 1e7) / 1e7 || 0
    },
    vr = function(t, r) {
        var e = r.charAt(0),
            i = parseFloat(r.substr(2));
        return t = parseFloat(t), e === "+" ? t + i : e === "-" ? t - i : e === "*" ? t * i : t / i
    },
    Yc = function(t, r) {
        for (var e = r.length, i = 0; t.indexOf(r[i]) < 0 && ++i < e;);
        return i < e
    },
    Un = function() {
        var t = ki.length,
            r = ki.slice(0),
            e, i;
        for (Rs = {}, ki.length = 0, e = 0; e < t; e++) i = r[e], i && i._lazy && (i.render(i._lazy[0], i._lazy[1], !0)._lazy = 0)
    },
    La = function(t, r, e, i) {
        ki.length && !Xt && Un(), t.render(r, e, Xt && r < 0 && (t._initted || t._startAt)), ki.length && !Xt && Un()
    },
    Ia = function(t) {
        var r = parseFloat(t);
        return (r || r === 0) && (t + "").match(Ca).length < 2 ? r : Dt(t) ? t.trim() : t
    },
    Ba = function(t) {
        return t
    },
    Re = function(t, r) {
        for (var e in r) e in t || (t[e] = r[e]);
        return t
    },
    Nc = function(t) {
        return function(r, e) {
            for (var i in e) i in r || i === "duration" && t || i === "ease" || (r[i] = e[i])
        }
    },
    Qi = function(t, r) {
        for (var e in r) t[e] = r[e];
        return t
    },
    Mo = function o(t, r) {
        for (var e in r) e !== "__proto__" && e !== "constructor" && e !== "prototype" && (t[e] = ti(r[e]) ? o(t[e] || (t[e] = {}), r[e]) : r[e]);
        return t
    },
    qn = function(t, r) {
        var e = {},
            i;
        for (i in t) i in r || (e[i] = t[i]);
        return e
    },
    jr = function(t) {
        var r = t.parent || ct,
            e = t.keyframes ? Nc(Ut(t.keyframes)) : Re;
        if (oe(t.inherit))
            for (; r;) e(t, r.vars.defaults), r = r.parent || r._dp;
        return t
    },
    Wc = function(t, r) {
        for (var e = t.length, i = e === r.length; i && e-- && t[e] === r[e];);
        return e < 0
    },
    za = function(t, r, e, i, n) {
        var s = t[i],
            a;
        if (n)
            for (a = r[n]; s && s[n] > a;) s = s._prev;
        return s ? (r._next = s._next, s._next = r) : (r._next = t[e], t[e] = r), r._next ? r._next._prev = r : t[i] = r, r._prev = s, r.parent = r._dp = t, r
    },
    ns = function(t, r, e, i) {
        e === void 0 && (e = "_first"), i === void 0 && (i = "_last");
        var n = r._prev,
            s = r._next;
        n ? n._next = s : t[e] === r && (t[e] = s), s ? s._prev = n : t[i] === r && (t[i] = n), r._next = r._prev = r.parent = null
    },
    Ci = function(t, r) {
        t.parent && (!r || t.parent.autoRemoveChildren) && t.parent.remove && t.parent.remove(t), t._act = 0
    },
    qi = function(t, r) {
        if (t && (!r || r._end > t._dur || r._start < 0))
            for (var e = t; e;) e._dirty = 1, e = e.parent;
        return t
    },
    Hc = function(t) {
        for (var r = t.parent; r && r.parent;) r._dirty = 1, r.totalDuration(), r = r.parent;
        return t
    },
    Ls = function(t, r, e, i) {
        return t._startAt && (Xt ? t._startAt.revert(Ln) : t.vars.immediateRender && !t.vars.autoRevert || t._startAt.render(r, !0, i))
    },
    Vc = function o(t) {
        return !t || t._ts && o(t.parent)
    },
    Do = function(t) {
        return t._repeat ? Er(t._tTime, t = t.duration() + t._rDelay) * t : 0
    },
    Er = function(t, r) {
        var e = Math.floor(t /= r);
        return t && e === t ? e - 1 : e
    },
    $n = function(t, r) {
        return (t - r._start) * r._ts + (r._ts >= 0 ? 0 : r._dirty ? r.totalDuration() : r._tDur)
    },
    ss = function(t) {
        return t._end = Mt(t._start + (t._tDur / Math.abs(t._ts || t._rts || rt) || 0))
    },
    os = function(t, r) {
        var e = t._dp;
        return e && e.smoothChildTiming && t._ts && (t._start = Mt(e._time - (t._ts > 0 ? r / t._ts : ((t._dirty ? t.totalDuration() : t._tDur) - r) / -t._ts)), ss(t), e._dirty || qi(e, t)), t
    },
    Fa = function(t, r) {
        var e;
        if ((r._time || !r._dur && r._initted || r._start < t._time && (r._dur || !r.add)) && (e = $n(t.rawTime(), r), (!r._dur || gn(0, r.totalDuration(), e) - r._tTime > rt) && r.render(e, !0)), qi(t, r)._dp && t._initted && t._time >= t._dur && t._ts) {
            if (t._dur < t.duration())
                for (e = t; e._dp;) e.rawTime() >= 0 && e.totalTime(e._tTime), e = e._dp;
            t._zTime = -rt
        }
    },
    Ge = function(t, r, e, i) {
        return r.parent && Ci(r), r._start = Mt((fi(e) ? e : e || t !== ct ? Oe(t, e, r) : t._time) + r._delay), r._end = Mt(r._start + (r.totalDuration() / Math.abs(r.timeScale()) || 0)), za(t, r, "_first", "_last", t._sort ? "_start" : 0), Is(r) || (t._recent = r), i || Fa(t, r), t._ts < 0 && os(t, t._tTime), t
    },
    Ya = function(t, r) {
        return (Te.ScrollTrigger || oo("scrollTrigger", r)) && Te.ScrollTrigger.create(r, t)
    },
    Na = function(t, r, e, i, n) {
        if (fo(t, r, n), !t._initted) return 1;
        if (!e && t._pt && !Xt && (t._dur && t.vars.lazy !== !1 || !t._dur && t.vars.lazy) && Da !== ye.frame) return ki.push(t), t._lazy = [n, i], 1
    },
    Xc = function o(t) {
        var r = t.parent;
        return r && r._ts && r._initted && !r._lock && (r.rawTime() < 0 || o(r))
    },
    Is = function(t) {
        var r = t.data;
        return r === "isFromStart" || r === "isStart"
    },
    Uc = function(t, r, e, i) {
        var n = t.ratio,
            s = r < 0 || !r && (!t._start && Xc(t) && !(!t._initted && Is(t)) || (t._ts < 0 || t._dp._ts < 0) && !Is(t)) ? 0 : 1,
            a = t._rDelay,
            l = 0,
            c, u, h;
        if (a && t._repeat && (l = gn(0, t._tDur, r), u = Er(l, a), t._yoyo && u & 1 && (s = 1 - s), u !== Er(t._tTime, a) && (n = 1 - s, t.vars.repeatRefresh && t._initted && t.invalidate())), s !== n || Xt || i || t._zTime === rt || !r && t._zTime) {
            if (!t._initted && Na(t, r, i, e, l)) return;
            for (h = t._zTime, t._zTime = r || (e ? rt : 0), e || (e = r && !h), t.ratio = s, t._from && (s = 1 - s), t._time = 0, t._tTime = l, c = t._pt; c;) c.r(s, c.d), c = c._next;
            r < 0 && Ls(t, r, e, !0), t._onUpdate && !e && be(t, "onUpdate"), l && t._repeat && !e && t.parent && be(t, "onRepeat"), (r >= t._tDur || r < 0) && t.ratio === s && (s && Ci(t, 1), !e && !Xt && (be(t, s ? "onComplete" : "onReverseComplete", !0), t._prom && t._prom()))
        } else t._zTime || (t._zTime = r)
    },
    qc = function(t, r, e) {
        var i;
        if (e > r)
            for (i = t._first; i && i._start <= e;) {
                if (i.data === "isPause" && i._start > r) return i;
                i = i._next
            } else
                for (i = t._last; i && i._start >= e;) {
                    if (i.data === "isPause" && i._start < r) return i;
                    i = i._prev
                }
    },
    Or = function(t, r, e, i) {
        var n = t._repeat,
            s = Mt(r) || 0,
            a = t._tTime / t._tDur;
        return a && !i && (t._time *= s / t._dur), t._dur = s, t._tDur = n ? n < 0 ? 1e10 : Mt(s * (n + 1) + t._rDelay * n) : s, a > 0 && !i && os(t, t._tTime = t._tDur * a), t.parent && ss(t), e || qi(t.parent, t), t
    },
    Ro = function(t) {
        return t instanceof ee ? qi(t) : Or(t, t._dur)
    },
    $c = {
        _start: 0,
        endTime: ln,
        totalDuration: ln
    },
    Oe = function o(t, r, e) {
        var i = t.labels,
            n = t._recent || $c,
            s = t.duration() >= Ae ? n.endTime(!1) : t._dur,
            a, l, c;
        return Dt(r) && (isNaN(r) || r in i) ? (l = r.charAt(0), c = r.substr(-1) === "%", a = r.indexOf("="), l === "<" || l === ">" ? (a >= 0 && (r = r.replace(/=/, "")), (l === "<" ? n._start : n.endTime(n._repeat >= 0)) + (parseFloat(r.substr(1)) || 0) * (c ? (a < 0 ? n : e).totalDuration() / 100 : 1)) : a < 0 ? (r in i || (i[r] = s), i[r]) : (l = parseFloat(r.charAt(a - 1) + r.substr(a + 1)), c && e && (l = l / 100 * (Ut(e) ? e[0] : e).totalDuration()), a > 1 ? o(t, r.substr(0, a - 1), e) + l : s + l)) : r == null ? s : +r
    },
    Gr = function(t, r, e) {
        var i = fi(r[1]),
            n = (i ? 2 : 1) + (t < 2 ? 0 : 1),
            s = r[n],
            a, l;
        if (i && (s.duration = r[1]), s.parent = e, t) {
            for (a = s, l = e; l && !("immediateRender" in a);) a = l.vars.defaults || {}, l = oe(l.vars.inherit) && l.parent;
            s.immediateRender = oe(a.immediateRender), t < 2 ? s.runBackwards = 1 : s.startAt = r[n - 1]
        }
        return new Tt(r[0], s, r[n + 1])
    },
    Mi = function(t, r) {
        return t || t === 0 ? r(t) : r
    },
    gn = function(t, r, e) {
        return e < t ? t : e > r ? r : e
    },
    Vt = function(t, r) {
        return !Dt(t) || !(r = Bc.exec(t)) ? "" : r[1]
    },
    jc = function(t, r, e) {
        return Mi(e, function(i) {
            return gn(t, r, i)
        })
    },
    Bs = [].slice,
    Wa = function(t, r) {
        return t && ti(t) && "length" in t && (!r && !t.length || t.length - 1 in t && ti(t[0])) && !t.nodeType && t !== $e
    },
    Gc = function(t, r, e) {
        return e === void 0 && (e = []), t.forEach(function(i) {
            var n;
            return Dt(i) && !r || Wa(i, 1) ? (n = e).push.apply(n, Me(i)) : e.push(i)
        }) || e
    },
    Me = function(t, r, e) {
        return ot && !r && ot.selector ? ot.selector(t) : Dt(t) && !e && (Ds || !Cr()) ? Bs.call((r || so).querySelectorAll(t), 0) : Ut(t) ? Gc(t, e) : Wa(t) ? Bs.call(t, 0) : t ? [t] : []
    },
    zs = function(t) {
        return t = Me(t)[0] || an("Invalid scope") || {},
            function(r) {
                var e = t.current || t.nativeElement || t;
                return Me(r, e.querySelectorAll ? e : e === t ? an("Invalid scope") || so.createElement("div") : t)
            }
    },
    Ha = function(t) {
        return t.sort(function() {
            return .5 - Math.random()
        })
    },
    Va = function(t) {
        if (dt(t)) return t;
        var r = ti(t) ? t : {
                each: t
            },
            e = $i(r.ease),
            i = r.from || 0,
            n = parseFloat(r.base) || 0,
            s = {},
            a = i > 0 && i < 1,
            l = isNaN(i) || a,
            c = r.axis,
            u = i,
            h = i;
        return Dt(i) ? u = h = {
                center: .5,
                edges: .5,
                end: 1
            } [i] || 0 : !a && l && (u = i[0], h = i[1]),
            function(d, f, _) {
                var p = (_ || r).length,
                    v = s[p],
                    T, b, E, g, y, S, w, C, k;
                if (!v) {
                    if (k = r.grid === "auto" ? 0 : (r.grid || [1, Ae])[1], !k) {
                        for (w = -Ae; w < (w = _[k++].getBoundingClientRect().left) && k < p;);
                        k < p && k--
                    }
                    for (v = s[p] = [], T = l ? Math.min(k, p) * u - .5 : i % k, b = k === Ae ? 0 : l ? p * h / k - .5 : i / k | 0, w = 0, C = Ae, S = 0; S < p; S++) E = S % k - T, g = b - (S / k | 0), v[S] = y = c ? Math.abs(c === "y" ? g : E) : Sa(E * E + g * g), y > w && (w = y), y < C && (C = y);
                    i === "random" && Ha(v), v.max = w - C, v.min = C, v.v = p = (parseFloat(r.amount) || parseFloat(r.each) * (k > p ? p - 1 : c ? c === "y" ? p / k : k : Math.max(k, p / k)) || 0) * (i === "edges" ? -1 : 1), v.b = p < 0 ? n - p : n, v.u = Vt(r.amount || r.each) || 0, e = e && p < 0 ? Ja(e) : e
                }
                return p = (v[d] - v.min) / v.max || 0, Mt(v.b + (e ? e(p) : p) * v.v) + v.u
            }
    },
    Fs = function(t) {
        var r = Math.pow(10, ((t + "").split(".")[1] || "").length);
        return function(e) {
            var i = Mt(Math.round(parseFloat(e) / t) * t * r);
            return (i - i % 1) / r + (fi(e) ? 0 : Vt(e))
        }
    },
    Xa = function(t, r) {
        var e = Ut(t),
            i, n;
        return !e && ti(t) && (i = e = t.radius || Ae, t.values ? (t = Me(t.values), (n = !fi(t[0])) && (i *= i)) : t = Fs(t.increment)), Mi(r, e ? dt(t) ? function(s) {
            return n = t(s), Math.abs(n - s) <= i ? n : s
        } : function(s) {
            for (var a = parseFloat(n ? s.x : s), l = parseFloat(n ? s.y : 0), c = Ae, u = 0, h = t.length, d, f; h--;) n ? (d = t[h].x - a, f = t[h].y - l, d = d * d + f * f) : d = Math.abs(t[h] - a), d < c && (c = d, u = h);
            return u = !i || c <= i ? t[u] : s, n || u === s || fi(s) ? u : u + Vt(s)
        } : Fs(t))
    },
    Ua = function(t, r, e, i) {
        return Mi(Ut(t) ? !r : e === !0 ? !!(e = 0) : !i, function() {
            return Ut(t) ? t[~~(Math.random() * t.length)] : (e = e || 1e-5) && (i = e < 1 ? Math.pow(10, (e + "").length - 2) : 1) && Math.floor(Math.round((t - e / 2 + Math.random() * (r - t + e * .99)) / e) * e * i) / i
        })
    },
    Kc = function() {
        for (var t = arguments.length, r = new Array(t), e = 0; e < t; e++) r[e] = arguments[e];
        return function(i) {
            return r.reduce(function(n, s) {
                return s(n)
            }, i)
        }
    },
    Zc = function(t, r) {
        return function(e) {
            return t(parseFloat(e)) + (r || Vt(e))
        }
    },
    Qc = function(t, r, e) {
        return $a(t, r, 0, 1, e)
    },
    qa = function(t, r, e) {
        return Mi(e, function(i) {
            return t[~~r(i)]
        })
    },
    Jc = function o(t, r, e) {
        var i = r - t;
        return Ut(t) ? qa(t, o(0, t.length), r) : Mi(e, function(n) {
            return (i + (n - t) % i) % i + t
        })
    },
    tu = function o(t, r, e) {
        var i = r - t,
            n = i * 2;
        return Ut(t) ? qa(t, o(0, t.length - 1), r) : Mi(e, function(s) {
            return s = (n + (s - t) % n) % n || 0, t + (s > i ? n - s : s)
        })
    },
    cn = function(t) {
        for (var r = 0, e = "", i, n, s, a; ~(i = t.indexOf("random(", r));) s = t.indexOf(")", i), a = t.charAt(i + 7) === "[", n = t.substr(i + 7, s - i - 7).match(a ? Ca : Ms), e += t.substr(r, i - r) + Ua(a ? n : +n[0], a ? 0 : +n[1], +n[2] || 1e-5), r = s + 1;
        return e + t.substr(r, t.length - r)
    },
    $a = function(t, r, e, i, n) {
        var s = r - t,
            a = i - e;
        return Mi(n, function(l) {
            return e + ((l - t) / s * a || 0)
        })
    },
    eu = function o(t, r, e, i) {
        var n = isNaN(t + r) ? 0 : function(f) {
            return (1 - f) * t + f * r
        };
        if (!n) {
            var s = Dt(t),
                a = {},
                l, c, u, h, d;
            if (e === !0 && (i = 1) && (e = null), s) t = {
                p: t
            }, r = {
                p: r
            };
            else if (Ut(t) && !Ut(r)) {
                for (u = [], h = t.length, d = h - 2, c = 1; c < h; c++) u.push(o(t[c - 1], t[c]));
                h--, n = function(_) {
                    _ *= h;
                    var p = Math.min(d, ~~_);
                    return u[p](_ - p)
                }, e = r
            } else i || (t = Qi(Ut(t) ? [] : {}, t));
            if (!u) {
                for (l in r) uo.call(a, t, l, "get", r[l]);
                n = function(_) {
                    return _o(_, a) || (s ? t.p : t)
                }
            }
        }
        return Mi(e, n)
    },
    Lo = function(t, r, e) {
        var i = t.labels,
            n = Ae,
            s, a, l;
        for (s in i) a = i[s] - r, a < 0 == !!e && a && n > (a = Math.abs(a)) && (l = s, n = a);
        return l
    },
    be = function(t, r, e) {
        var i = t.vars,
            n = i[r],
            s = ot,
            a = t._ctx,
            l, c, u;
        if (n) return l = i[r + "Params"], c = i.callbackScope || t, e && ki.length && Un(), a && (ot = a), u = l ? n.apply(c, l) : n.call(c), ot = s, u
    },
    Wr = function(t) {
        return Ci(t), t.scrollTrigger && t.scrollTrigger.kill(!!Xt), t.progress() < 1 && be(t, "onInterrupt"), t
    },
    _r, ja = [],
    Ga = function(t) {
        if (t)
            if (t = !t.name && t.default || t, no() || t.headless) {
                var r = t.name,
                    e = dt(t),
                    i = r && !e && t.init ? function() {
                        this._props = []
                    } : t,
                    n = {
                        init: ln,
                        render: _o,
                        add: uo,
                        kill: mu,
                        modifier: gu,
                        rawVars: 0
                    },
                    s = {
                        targetTest: 0,
                        get: 0,
                        getSetter: po,
                        aliases: {},
                        register: 0
                    };
                if (Cr(), t !== i) {
                    if (me[r]) return;
                    Re(i, Re(qn(t, n), s)), Qi(i.prototype, Qi(n, qn(t, s))), me[i.prop = r] = i, t.targetTest && (In.push(i), ao[r] = 1), r = (r === "css" ? "CSS" : r.charAt(0).toUpperCase() + r.substr(1)) + "Plugin"
                }
                Ma(r, i), t.register && t.register(ue, i, le)
            } else ja.push(t)
    },
    it = 255,
    Hr = {
        aqua: [0, it, it],
        lime: [0, it, 0],
        silver: [192, 192, 192],
        black: [0, 0, 0],
        maroon: [128, 0, 0],
        teal: [0, 128, 128],
        blue: [0, 0, it],
        navy: [0, 0, 128],
        white: [it, it, it],
        olive: [128, 128, 0],
        yellow: [it, it, 0],
        orange: [it, 165, 0],
        gray: [128, 128, 128],
        purple: [128, 0, 128],
        green: [0, 128, 0],
        red: [it, 0, 0],
        pink: [it, 192, 203],
        cyan: [0, it, it],
        transparent: [it, it, it, 0]
    },
    ps = function(t, r, e) {
        return t += t < 0 ? 1 : t > 1 ? -1 : 0, (t * 6 < 1 ? r + (e - r) * t * 6 : t < .5 ? e : t * 3 < 2 ? r + (e - r) * (2 / 3 - t) * 6 : r) * it + .5 | 0
    },
    Ka = function(t, r, e) {
        var i = t ? fi(t) ? [t >> 16, t >> 8 & it, t & it] : 0 : Hr.black,
            n, s, a, l, c, u, h, d, f, _;
        if (!i) {
            if (t.substr(-1) === "," && (t = t.substr(0, t.length - 1)), Hr[t]) i = Hr[t];
            else if (t.charAt(0) === "#") {
                if (t.length < 6 && (n = t.charAt(1), s = t.charAt(2), a = t.charAt(3), t = "#" + n + n + s + s + a + a + (t.length === 5 ? t.charAt(4) + t.charAt(4) : "")), t.length === 9) return i = parseInt(t.substr(1, 6), 16), [i >> 16, i >> 8 & it, i & it, parseInt(t.substr(7), 16) / 255];
                t = parseInt(t.substr(1), 16), i = [t >> 16, t >> 8 & it, t & it]
            } else if (t.substr(0, 3) === "hsl") {
                if (i = _ = t.match(Ms), !r) l = +i[0] % 360 / 360, c = +i[1] / 100, u = +i[2] / 100, s = u <= .5 ? u * (c + 1) : u + c - u * c, n = u * 2 - s, i.length > 3 && (i[3] *= 1), i[0] = ps(l + 1 / 3, n, s), i[1] = ps(l, n, s), i[2] = ps(l - 1 / 3, n, s);
                else if (~t.indexOf("=")) return i = t.match(Ea), e && i.length < 4 && (i[3] = 1), i
            } else i = t.match(Ms) || Hr.transparent;
            i = i.map(Number)
        }
        return r && !_ && (n = i[0] / it, s = i[1] / it, a = i[2] / it, h = Math.max(n, s, a), d = Math.min(n, s, a), u = (h + d) / 2, h === d ? l = c = 0 : (f = h - d, c = u > .5 ? f / (2 - h - d) : f / (h + d), l = h === n ? (s - a) / f + (s < a ? 6 : 0) : h === s ? (a - n) / f + 2 : (n - s) / f + 4, l *= 60), i[0] = ~~(l + .5), i[1] = ~~(c * 100 + .5), i[2] = ~~(u * 100 + .5)), e && i.length < 4 && (i[3] = 1), i
    },
    Za = function(t) {
        var r = [],
            e = [],
            i = -1;
        return t.split(Ei).forEach(function(n) {
            var s = n.match(pr) || [];
            r.push.apply(r, s), e.push(i += s.length + 1)
        }), r.c = e, r
    },
    Io = function(t, r, e) {
        var i = "",
            n = (t + i).match(Ei),
            s = r ? "hsla(" : "rgba(",
            a = 0,
            l, c, u, h;
        if (!n) return t;
        if (n = n.map(function(d) {
                return (d = Ka(d, r, 1)) && s + (r ? d[0] + "," + d[1] + "%," + d[2] + "%," + d[3] : d.join(",")) + ")"
            }), e && (u = Za(t), l = e.c, l.join(i) !== u.c.join(i)))
            for (c = t.replace(Ei, "1").split(pr), h = c.length - 1; a < h; a++) i += c[a] + (~l.indexOf(a) ? n.shift() || s + "0,0,0,0)" : (u.length ? u : n.length ? n : e).shift());
        if (!c)
            for (c = t.split(Ei), h = c.length - 1; a < h; a++) i += c[a] + n[a];
        return i + c[h]
    },
    Ei = function() {
        var o = "(?:\\b(?:(?:rgb|rgba|hsl|hsla)\\(.+?\\))|\\B#(?:[0-9a-f]{3,4}){1,2}\\b",
            t;
        for (t in Hr) o += "|" + t + "\\b";
        return new RegExp(o + ")", "gi")
    }(),
    iu = /hsl[a]?\(/,
    Qa = function(t) {
        var r = t.join(" "),
            e;
        if (Ei.lastIndex = 0, Ei.test(r)) return e = iu.test(r), t[1] = Io(t[1], e), t[0] = Io(t[0], e, Za(t[1])), !0
    },
    un, ye = function() {
        var o = Date.now,
            t = 500,
            r = 33,
            e = o(),
            i = e,
            n = 1e3 / 240,
            s = n,
            a = [],
            l, c, u, h, d, f, _ = function p(v) {
                var T = o() - i,
                    b = v === !0,
                    E, g, y, S;
                if ((T > t || T < 0) && (e += T - r), i += T, y = i - e, E = y - s, (E > 0 || b) && (S = ++h.frame, d = y - h.time * 1e3, h.time = y = y / 1e3, s += E + (E >= n ? 4 : n - E), g = 1), b || (l = c(p)), g)
                    for (f = 0; f < a.length; f++) a[f](y, d, S, v)
            };
        return h = {
            time: 0,
            frame: 0,
            tick: function() {
                _(!0)
            },
            deltaRatio: function(v) {
                return d / (1e3 / (v || 60))
            },
            wake: function() {
                Pa && (!Ds && no() && ($e = Ds = window, so = $e.document || {}, Te.gsap = ue, ($e.gsapVersions || ($e.gsapVersions = [])).push(ue.version), Aa(Xn || $e.GreenSockGlobals || !$e.gsap && $e || {}), ja.forEach(Ga)), u = typeof requestAnimationFrame < "u" && requestAnimationFrame, l && h.sleep(), c = u || function(v) {
                    return setTimeout(v, s - h.time * 1e3 + 1 | 0)
                }, un = 1, _(2))
            },
            sleep: function() {
                (u ? cancelAnimationFrame : clearTimeout)(l), un = 0, c = ln
            },
            lagSmoothing: function(v, T) {
                t = v || 1 / 0, r = Math.min(T || 33, t)
            },
            fps: function(v) {
                n = 1e3 / (v || 240), s = h.time * 1e3 + n
            },
            add: function(v, T, b) {
                var E = T ? function(g, y, S, w) {
                    v(g, y, S, w), h.remove(E)
                } : v;
                return h.remove(v), a[b ? "unshift" : "push"](E), Cr(), E
            },
            remove: function(v, T) {
                ~(T = a.indexOf(v)) && a.splice(T, 1) && f >= T && f--
            },
            _listeners: a
        }, h
    }(),
    Cr = function() {
        return !un && ye.wake()
    },
    K = {},
    ru = /^[\d.\-M][\d.\-,\s]/,
    nu = /["']/g,
    su = function(t) {
        for (var r = {}, e = t.substr(1, t.length - 3).split(":"), i = e[0], n = 1, s = e.length, a, l, c; n < s; n++) l = e[n], a = n !== s - 1 ? l.lastIndexOf(",") : l.length, c = l.substr(0, a), r[i] = isNaN(c) ? c.replace(nu, "").trim() : +c, i = l.substr(a + 1).trim();
        return r
    },
    ou = function(t) {
        var r = t.indexOf("(") + 1,
            e = t.indexOf(")"),
            i = t.indexOf("(", r);
        return t.substring(r, ~i && i < e ? t.indexOf(")", e + 1) : e)
    },
    au = function(t) {
        var r = (t + "").split("("),
            e = K[r[0]];
        return e && r.length > 1 && e.config ? e.config.apply(null, ~t.indexOf("{") ? [su(r[1])] : ou(t).split(",").map(Ia)) : K._CE && ru.test(t) ? K._CE("", t) : e
    },
    Ja = function(t) {
        return function(r) {
            return 1 - t(1 - r)
        }
    },
    tl = function o(t, r) {
        for (var e = t._first, i; e;) e instanceof ee ? o(e, r) : e.vars.yoyoEase && (!e._yoyo || !e._repeat) && e._yoyo !== r && (e.timeline ? o(e.timeline, r) : (i = e._ease, e._ease = e._yEase, e._yEase = i, e._yoyo = r)), e = e._next
    },
    $i = function(t, r) {
        return t && (dt(t) ? t : K[t] || au(t)) || r
    },
    rr = function(t, r, e, i) {
        e === void 0 && (e = function(l) {
            return 1 - r(1 - l)
        }), i === void 0 && (i = function(l) {
            return l < .5 ? r(l * 2) / 2 : 1 - r((1 - l) * 2) / 2
        });
        var n = {
                easeIn: r,
                easeOut: e,
                easeInOut: i
            },
            s;
        return ae(t, function(a) {
            K[a] = Te[a] = n, K[s = a.toLowerCase()] = e;
            for (var l in n) K[s + (l === "easeIn" ? ".in" : l === "easeOut" ? ".out" : ".inOut")] = K[a + "." + l] = n[l]
        }), n
    },
    el = function(t) {
        return function(r) {
            return r < .5 ? (1 - t(1 - r * 2)) / 2 : .5 + t((r - .5) * 2) / 2
        }
    },
    _s = function o(t, r, e) {
        var i = r >= 1 ? r : 1,
            n = (e || (t ? .3 : .45)) / (r < 1 ? r : 1),
            s = n / As * (Math.asin(1 / i) || 0),
            a = function(u) {
                return u === 1 ? 1 : i * Math.pow(2, -10 * u) * Ic((u - s) * n) + 1
            },
            l = t === "out" ? a : t === "in" ? function(c) {
                return 1 - a(1 - c)
            } : el(a);
        return n = As / n, l.config = function(c, u) {
            return o(t, c, u)
        }, l
    },
    gs = function o(t, r) {
        r === void 0 && (r = 1.70158);
        var e = function(s) {
                return s ? --s * s * ((r + 1) * s + r) + 1 : 0
            },
            i = t === "out" ? e : t === "in" ? function(n) {
                return 1 - e(1 - n)
            } : el(e);
        return i.config = function(n) {
            return o(t, n)
        }, i
    };
ae("Linear,Quad,Cubic,Quart,Quint,Strong", function(o, t) {
    var r = t < 5 ? t + 1 : t;
    rr(o + ",Power" + (r - 1), t ? function(e) {
        return Math.pow(e, r)
    } : function(e) {
        return e
    }, function(e) {
        return 1 - Math.pow(1 - e, r)
    }, function(e) {
        return e < .5 ? Math.pow(e * 2, r) / 2 : 1 - Math.pow((1 - e) * 2, r) / 2
    })
});
K.Linear.easeNone = K.none = K.Linear.easeIn;
rr("Elastic", _s("in"), _s("out"), _s());
(function(o, t) {
    var r = 1 / t,
        e = 2 * r,
        i = 2.5 * r,
        n = function(a) {
            return a < r ? o * a * a : a < e ? o * Math.pow(a - 1.5 / t, 2) + .75 : a < i ? o * (a -= 2.25 / t) * a + .9375 : o * Math.pow(a - 2.625 / t, 2) + .984375
        };
    rr("Bounce", function(s) {
        return 1 - n(1 - s)
    }, n)
})(7.5625, 2.75);
rr("Expo", function(o) {
    return o ? Math.pow(2, 10 * (o - 1)) : 0
});
rr("Circ", function(o) {
    return -(Sa(1 - o * o) - 1)
});
rr("Sine", function(o) {
    return o === 1 ? 1 : -Lc(o * Dc) + 1
});
rr("Back", gs("in"), gs("out"), gs());
K.SteppedEase = K.steps = Te.SteppedEase = {
    config: function(t, r) {
        t === void 0 && (t = 1);
        var e = 1 / t,
            i = t + (r ? 0 : 1),
            n = r ? 1 : 0,
            s = 1 - rt;
        return function(a) {
            return ((i * gn(0, s, a) | 0) + n) * e
        }
    }
};
kr.ease = K["quad.out"];
ae("onComplete,onUpdate,onStart,onRepeat,onReverseComplete,onInterrupt", function(o) {
    return lo += o + "," + o + "Params,"
});
var il = function(t, r) {
        this.id = Rc++, t._gsap = this, this.target = t, this.harness = r, this.get = r ? r.get : Ra, this.set = r ? r.getSetter : po
    },
    fn = function() {
        function o(r) {
            this.vars = r, this._delay = +r.delay || 0, (this._repeat = r.repeat === 1 / 0 ? -2 : r.repeat || 0) && (this._rDelay = r.repeatDelay || 0, this._yoyo = !!r.yoyo || !!r.yoyoEase), this._ts = 1, Or(this, +r.duration, 1, 1), this.data = r.data, ot && (this._ctx = ot, ot.data.push(this)), un || ye.wake()
        }
        var t = o.prototype;
        return t.delay = function(e) {
            return e || e === 0 ? (this.parent && this.parent.smoothChildTiming && this.startTime(this._start + e - this._delay), this._delay = e, this) : this._delay
        }, t.duration = function(e) {
            return arguments.length ? this.totalDuration(this._repeat > 0 ? e + (e + this._rDelay) * this._repeat : e) : this.totalDuration() && this._dur
        }, t.totalDuration = function(e) {
            return arguments.length ? (this._dirty = 0, Or(this, this._repeat < 0 ? e : (e - this._repeat * this._rDelay) / (this._repeat + 1))) : this._tDur
        }, t.totalTime = function(e, i) {
            if (Cr(), !arguments.length) return this._tTime;
            var n = this._dp;
            if (n && n.smoothChildTiming && this._ts) {
                for (os(this, e), !n._dp || n.parent || Fa(n, this); n && n.parent;) n.parent._time !== n._start + (n._ts >= 0 ? n._tTime / n._ts : (n.totalDuration() - n._tTime) / -n._ts) && n.totalTime(n._tTime, !0), n = n.parent;
                !this.parent && this._dp.autoRemoveChildren && (this._ts > 0 && e < this._tDur || this._ts < 0 && e > 0 || !this._tDur && !e) && Ge(this._dp, this, this._start - this._delay)
            }
            return (this._tTime !== e || !this._dur && !i || this._initted && Math.abs(this._zTime) === rt || !e && !this._initted && (this.add || this._ptLookup)) && (this._ts || (this._pTime = e), La(this, e, i)), this
        }, t.time = function(e, i) {
            return arguments.length ? this.totalTime(Math.min(this.totalDuration(), e + Do(this)) % (this._dur + this._rDelay) || (e ? this._dur : 0), i) : this._time
        }, t.totalProgress = function(e, i) {
            return arguments.length ? this.totalTime(this.totalDuration() * e, i) : this.totalDuration() ? Math.min(1, this._tTime / this._tDur) : this.rawTime() > 0 ? 1 : 0
        }, t.progress = function(e, i) {
            return arguments.length ? this.totalTime(this.duration() * (this._yoyo && !(this.iteration() & 1) ? 1 - e : e) + Do(this), i) : this.duration() ? Math.min(1, this._time / this._dur) : this.rawTime() > 0 ? 1 : 0
        }, t.iteration = function(e, i) {
            var n = this.duration() + this._rDelay;
            return arguments.length ? this.totalTime(this._time + (e - 1) * n, i) : this._repeat ? Er(this._tTime, n) + 1 : 1
        }, t.timeScale = function(e, i) {
            if (!arguments.length) return this._rts === -rt ? 0 : this._rts;
            if (this._rts === e) return this;
            var n = this.parent && this._ts ? $n(this.parent._time, this) : this._tTime;
            return this._rts = +e || 0, this._ts = this._ps || e === -rt ? 0 : this._rts, this.totalTime(gn(-Math.abs(this._delay), this._tDur, n), i !== !1), ss(this), Hc(this)
        }, t.paused = function(e) {
            return arguments.length ? (this._ps !== e && (this._ps = e, e ? (this._pTime = this._tTime || Math.max(-this._delay, this.rawTime()), this._ts = this._act = 0) : (Cr(), this._ts = this._rts, this.totalTime(this.parent && !this.parent.smoothChildTiming ? this.rawTime() : this._tTime || this._pTime, this.progress() === 1 && Math.abs(this._zTime) !== rt && (this._tTime -= rt)))), this) : this._ps
        }, t.startTime = function(e) {
            if (arguments.length) {
                this._start = e;
                var i = this.parent || this._dp;
                return i && (i._sort || !this.parent) && Ge(i, this, e - this._delay), this
            }
            return this._start
        }, t.endTime = function(e) {
            return this._start + (oe(e) ? this.totalDuration() : this.duration()) / Math.abs(this._ts || 1)
        }, t.rawTime = function(e) {
            var i = this.parent || this._dp;
            return i ? e && (!this._ts || this._repeat && this._time && this.totalProgress() < 1) ? this._tTime % (this._dur + this._rDelay) : this._ts ? $n(i.rawTime(e), this) : this._tTime : this._tTime
        }, t.revert = function(e) {
            e === void 0 && (e = Fc);
            var i = Xt;
            return Xt = e, (this._initted || this._startAt) && (this.timeline && this.timeline.revert(e), this.totalTime(-.01, e.suppressEvents)), this.data !== "nested" && e.kill !== !1 && this.kill(), Xt = i, this
        }, t.globalTime = function(e) {
            for (var i = this, n = arguments.length ? e : i.rawTime(); i;) n = i._start + n / (Math.abs(i._ts) || 1), i = i._dp;
            return !this.parent && this._sat ? this._sat.globalTime(e) : n
        }, t.repeat = function(e) {
            return arguments.length ? (this._repeat = e === 1 / 0 ? -2 : e, Ro(this)) : this._repeat === -2 ? 1 / 0 : this._repeat
        }, t.repeatDelay = function(e) {
            if (arguments.length) {
                var i = this._time;
                return this._rDelay = e, Ro(this), i ? this.time(i) : this
            }
            return this._rDelay
        }, t.yoyo = function(e) {
            return arguments.length ? (this._yoyo = e, this) : this._yoyo
        }, t.seek = function(e, i) {
            return this.totalTime(Oe(this, e), oe(i))
        }, t.restart = function(e, i) {
            return this.play().totalTime(e ? -this._delay : 0, oe(i))
        }, t.play = function(e, i) {
            return e != null && this.seek(e, i), this.reversed(!1).paused(!1)
        }, t.reverse = function(e, i) {
            return e != null && this.seek(e || this.totalDuration(), i), this.reversed(!0).paused(!1)
        }, t.pause = function(e, i) {
            return e != null && this.seek(e, i), this.paused(!0)
        }, t.resume = function() {
            return this.paused(!1)
        }, t.reversed = function(e) {
            return arguments.length ? (!!e !== this.reversed() && this.timeScale(-this._rts || (e ? -rt : 0)), this) : this._rts < 0
        }, t.invalidate = function() {
            return this._initted = this._act = 0, this._zTime = -rt, this
        }, t.isActive = function() {
            var e = this.parent || this._dp,
                i = this._start,
                n;
            return !!(!e || this._ts && this._initted && e.isActive() && (n = e.rawTime(!0)) >= i && n < this.endTime(!0) - rt)
        }, t.eventCallback = function(e, i, n) {
            var s = this.vars;
            return arguments.length > 1 ? (i ? (s[e] = i, n && (s[e + "Params"] = n), e === "onUpdate" && (this._onUpdate = i)) : delete s[e], this) : s[e]
        }, t.then = function(e) {
            var i = this;
            return new Promise(function(n) {
                var s = dt(e) ? e : Ba,
                    a = function() {
                        var c = i.then;
                        i.then = null, dt(s) && (s = s(i)) && (s.then || s === i) && (i.then = c), n(s), i.then = c
                    };
                i._initted && i.totalProgress() === 1 && i._ts >= 0 || !i._tTime && i._ts < 0 ? a() : i._prom = a
            })
        }, t.kill = function() {
            Wr(this)
        }, o
    }();
Re(fn.prototype, {
    _time: 0,
    _start: 0,
    _end: 0,
    _tTime: 0,
    _tDur: 0,
    _dirty: 0,
    _repeat: 0,
    _yoyo: !1,
    parent: null,
    _initted: !1,
    _rDelay: 0,
    _ts: 1,
    _dp: 0,
    ratio: 0,
    _zTime: -rt,
    _prom: 0,
    _ps: !1,
    _rts: 1
});
var ee = function(o) {
    Ta(t, o);

    function t(e, i) {
        var n;
        return e === void 0 && (e = {}), n = o.call(this, e) || this, n.labels = {}, n.smoothChildTiming = !!e.smoothChildTiming, n.autoRemoveChildren = !!e.autoRemoveChildren, n._sort = oe(e.sortChildren), ct && Ge(e.parent || ct, si(n), i), e.reversed && n.reverse(), e.paused && n.paused(!0), e.scrollTrigger && Ya(si(n), e.scrollTrigger), n
    }
    var r = t.prototype;
    return r.to = function(i, n, s) {
        return Gr(0, arguments, this), this
    }, r.from = function(i, n, s) {
        return Gr(1, arguments, this), this
    }, r.fromTo = function(i, n, s, a) {
        return Gr(2, arguments, this), this
    }, r.set = function(i, n, s) {
        return n.duration = 0, n.parent = this, jr(n).repeatDelay || (n.repeat = 0), n.immediateRender = !!n.immediateRender, new Tt(i, n, Oe(this, s), 1), this
    }, r.call = function(i, n, s) {
        return Ge(this, Tt.delayedCall(0, i, n), s)
    }, r.staggerTo = function(i, n, s, a, l, c, u) {
        return s.duration = n, s.stagger = s.stagger || a, s.onComplete = c, s.onCompleteParams = u, s.parent = this, new Tt(i, s, Oe(this, l)), this
    }, r.staggerFrom = function(i, n, s, a, l, c, u) {
        return s.runBackwards = 1, jr(s).immediateRender = oe(s.immediateRender), this.staggerTo(i, n, s, a, l, c, u)
    }, r.staggerFromTo = function(i, n, s, a, l, c, u, h) {
        return a.startAt = s, jr(a).immediateRender = oe(a.immediateRender), this.staggerTo(i, n, a, l, c, u, h)
    }, r.render = function(i, n, s) {
        var a = this._time,
            l = this._dirty ? this.totalDuration() : this._tDur,
            c = this._dur,
            u = i <= 0 ? 0 : Mt(i),
            h = this._zTime < 0 != i < 0 && (this._initted || !c),
            d, f, _, p, v, T, b, E, g, y, S, w;
        if (this !== ct && u > l && i >= 0 && (u = l), u !== this._tTime || s || h) {
            if (a !== this._time && c && (u += this._time - a, i += this._time - a), d = u, g = this._start, E = this._ts, T = !E, h && (c || (a = this._zTime), (i || !n) && (this._zTime = i)), this._repeat) {
                if (S = this._yoyo, v = c + this._rDelay, this._repeat < -1 && i < 0) return this.totalTime(v * 100 + i, n, s);
                if (d = Mt(u % v), u === l ? (p = this._repeat, d = c) : (p = ~~(u / v), p && p === u / v && (d = c, p--), d > c && (d = c)), y = Er(this._tTime, v), !a && this._tTime && y !== p && this._tTime - y * v - this._dur <= 0 && (y = p), S && p & 1 && (d = c - d, w = 1), p !== y && !this._lock) {
                    var C = S && y & 1,
                        k = C === (S && p & 1);
                    if (p < y && (C = !C), a = C ? 0 : u % c ? c : u, this._lock = 1, this.render(a || (w ? 0 : Mt(p * v)), n, !c)._lock = 0, this._tTime = u, !n && this.parent && be(this, "onRepeat"), this.vars.repeatRefresh && !w && (this.invalidate()._lock = 1), a && a !== this._time || T !== !this._ts || this.vars.onRepeat && !this.parent && !this._act) return this;
                    if (c = this._dur, l = this._tDur, k && (this._lock = 2, a = C ? c : -1e-4, this.render(a, !0), this.vars.repeatRefresh && !w && this.invalidate()), this._lock = 0, !this._ts && !T) return this;
                    tl(this, w)
                }
            }
            if (this._hasPause && !this._forcing && this._lock < 2 && (b = qc(this, Mt(a), Mt(d)), b && (u -= d - (d = b._start))), this._tTime = u, this._time = d, this._act = !E, this._initted || (this._onUpdate = this.vars.onUpdate, this._initted = 1, this._zTime = i, a = 0), !a && d && !n && !p && (be(this, "onStart"), this._tTime !== u)) return this;
            if (d >= a && i >= 0)
                for (f = this._first; f;) {
                    if (_ = f._next, (f._act || d >= f._start) && f._ts && b !== f) {
                        if (f.parent !== this) return this.render(i, n, s);
                        if (f.render(f._ts > 0 ? (d - f._start) * f._ts : (f._dirty ? f.totalDuration() : f._tDur) + (d - f._start) * f._ts, n, s), d !== this._time || !this._ts && !T) {
                            b = 0, _ && (u += this._zTime = -rt);
                            break
                        }
                    }
                    f = _
                } else {
                    f = this._last;
                    for (var O = i < 0 ? i : d; f;) {
                        if (_ = f._prev, (f._act || O <= f._end) && f._ts && b !== f) {
                            if (f.parent !== this) return this.render(i, n, s);
                            if (f.render(f._ts > 0 ? (O - f._start) * f._ts : (f._dirty ? f.totalDuration() : f._tDur) + (O - f._start) * f._ts, n, s || Xt && (f._initted || f._startAt)), d !== this._time || !this._ts && !T) {
                                b = 0, _ && (u += this._zTime = O ? -rt : rt);
                                break
                            }
                        }
                        f = _
                    }
                }
            if (b && !n && (this.pause(), b.render(d >= a ? 0 : -rt)._zTime = d >= a ? 1 : -1, this._ts)) return this._start = g, ss(this), this.render(i, n, s);
            this._onUpdate && !n && be(this, "onUpdate", !0), (u === l && this._tTime >= this.totalDuration() || !u && a) && (g === this._start || Math.abs(E) !== Math.abs(this._ts)) && (this._lock || ((i || !c) && (u === l && this._ts > 0 || !u && this._ts < 0) && Ci(this, 1), !n && !(i < 0 && !a) && (u || a || !l) && (be(this, u === l && i >= 0 ? "onComplete" : "onReverseComplete", !0), this._prom && !(u < l && this.timeScale() > 0) && this._prom())))
        }
        return this
    }, r.add = function(i, n) {
        var s = this;
        if (fi(n) || (n = Oe(this, n, i)), !(i instanceof fn)) {
            if (Ut(i)) return i.forEach(function(a) {
                return s.add(a, n)
            }), this;
            if (Dt(i)) return this.addLabel(i, n);
            if (dt(i)) i = Tt.delayedCall(0, i);
            else return this
        }
        return this !== i ? Ge(this, i, n) : this
    }, r.getChildren = function(i, n, s, a) {
        i === void 0 && (i = !0), n === void 0 && (n = !0), s === void 0 && (s = !0), a === void 0 && (a = -Ae);
        for (var l = [], c = this._first; c;) c._start >= a && (c instanceof Tt ? n && l.push(c) : (s && l.push(c), i && l.push.apply(l, c.getChildren(!0, n, s)))), c = c._next;
        return l
    }, r.getById = function(i) {
        for (var n = this.getChildren(1, 1, 1), s = n.length; s--;)
            if (n[s].vars.id === i) return n[s]
    }, r.remove = function(i) {
        return Dt(i) ? this.removeLabel(i) : dt(i) ? this.killTweensOf(i) : (ns(this, i), i === this._recent && (this._recent = this._last), qi(this))
    }, r.totalTime = function(i, n) {
        return arguments.length ? (this._forcing = 1, !this._dp && this._ts && (this._start = Mt(ye.time - (this._ts > 0 ? i / this._ts : (this.totalDuration() - i) / -this._ts))), o.prototype.totalTime.call(this, i, n), this._forcing = 0, this) : this._tTime
    }, r.addLabel = function(i, n) {
        return this.labels[i] = Oe(this, n), this
    }, r.removeLabel = function(i) {
        return delete this.labels[i], this
    }, r.addPause = function(i, n, s) {
        var a = Tt.delayedCall(0, n || ln, s);
        return a.data = "isPause", this._hasPause = 1, Ge(this, a, Oe(this, i))
    }, r.removePause = function(i) {
        var n = this._first;
        for (i = Oe(this, i); n;) n._start === i && n.data === "isPause" && Ci(n), n = n._next
    }, r.killTweensOf = function(i, n, s) {
        for (var a = this.getTweensOf(i, s), l = a.length; l--;) wi !== a[l] && a[l].kill(i, n);
        return this
    }, r.getTweensOf = function(i, n) {
        for (var s = [], a = Me(i), l = this._first, c = fi(n), u; l;) l instanceof Tt ? Yc(l._targets, a) && (c ? (!wi || l._initted && l._ts) && l.globalTime(0) <= n && l.globalTime(l.totalDuration()) > n : !n || l.isActive()) && s.push(l) : (u = l.getTweensOf(a, n)).length && s.push.apply(s, u), l = l._next;
        return s
    }, r.tweenTo = function(i, n) {
        n = n || {};
        var s = this,
            a = Oe(s, i),
            l = n,
            c = l.startAt,
            u = l.onStart,
            h = l.onStartParams,
            d = l.immediateRender,
            f, _ = Tt.to(s, Re({
                ease: n.ease || "none",
                lazy: !1,
                immediateRender: !1,
                time: a,
                overwrite: "auto",
                duration: n.duration || Math.abs((a - (c && "time" in c ? c.time : s._time)) / s.timeScale()) || rt,
                onStart: function() {
                    if (s.pause(), !f) {
                        var v = n.duration || Math.abs((a - (c && "time" in c ? c.time : s._time)) / s.timeScale());
                        _._dur !== v && Or(_, v, 0, 1).render(_._time, !0, !0), f = 1
                    }
                    u && u.apply(_, h || [])
                }
            }, n));
        return d ? _.render(0) : _
    }, r.tweenFromTo = function(i, n, s) {
        return this.tweenTo(n, Re({
            startAt: {
                time: Oe(this, i)
            }
        }, s))
    }, r.recent = function() {
        return this._recent
    }, r.nextLabel = function(i) {
        return i === void 0 && (i = this._time), Lo(this, Oe(this, i))
    }, r.previousLabel = function(i) {
        return i === void 0 && (i = this._time), Lo(this, Oe(this, i), 1)
    }, r.currentLabel = function(i) {
        return arguments.length ? this.seek(i, !0) : this.previousLabel(this._time + rt)
    }, r.shiftChildren = function(i, n, s) {
        s === void 0 && (s = 0);
        for (var a = this._first, l = this.labels, c; a;) a._start >= s && (a._start += i, a._end += i), a = a._next;
        if (n)
            for (c in l) l[c] >= s && (l[c] += i);
        return qi(this)
    }, r.invalidate = function(i) {
        var n = this._first;
        for (this._lock = 0; n;) n.invalidate(i), n = n._next;
        return o.prototype.invalidate.call(this, i)
    }, r.clear = function(i) {
        i === void 0 && (i = !0);
        for (var n = this._first, s; n;) s = n._next, this.remove(n), n = s;
        return this._dp && (this._time = this._tTime = this._pTime = 0), i && (this.labels = {}), qi(this)
    }, r.totalDuration = function(i) {
        var n = 0,
            s = this,
            a = s._last,
            l = Ae,
            c, u, h;
        if (arguments.length) return s.timeScale((s._repeat < 0 ? s.duration() : s.totalDuration()) / (s.reversed() ? -i : i));
        if (s._dirty) {
            for (h = s.parent; a;) c = a._prev, a._dirty && a.totalDuration(), u = a._start, u > l && s._sort && a._ts && !s._lock ? (s._lock = 1, Ge(s, a, u - a._delay, 1)._lock = 0) : l = u, u < 0 && a._ts && (n -= u, (!h && !s._dp || h && h.smoothChildTiming) && (s._start += u / s._ts, s._time -= u, s._tTime -= u), s.shiftChildren(-u, !1, -1 / 0), l = 0), a._end > n && a._ts && (n = a._end), a = c;
            Or(s, s === ct && s._time > n ? s._time : n, 1, 1), s._dirty = 0
        }
        return s._tDur
    }, t.updateRoot = function(i) {
        if (ct._ts && (La(ct, $n(i, ct)), Da = ye.frame), ye.frame >= Ao) {
            Ao += xe.autoSleep || 120;
            var n = ct._first;
            if ((!n || !n._ts) && xe.autoSleep && ye._listeners.length < 2) {
                for (; n && !n._ts;) n = n._next;
                n || ye.sleep()
            }
        }
    }, t
}(fn);
Re(ee.prototype, {
    _lock: 0,
    _hasPause: 0,
    _forcing: 0
});
var lu = function(t, r, e, i, n, s, a) {
        var l = new le(this._pt, t, r, 0, 1, ll, null, n),
            c = 0,
            u = 0,
            h, d, f, _, p, v, T, b;
        for (l.b = e, l.e = i, e += "", i += "", (T = ~i.indexOf("random(")) && (i = cn(i)), s && (b = [e, i], s(b, t, r), e = b[0], i = b[1]), d = e.match(hs) || []; h = hs.exec(i);) _ = h[0], p = i.substring(c, h.index), f ? f = (f + 1) % 5 : p.substr(-5) === "rgba(" && (f = 1), _ !== d[u++] && (v = parseFloat(d[u - 1]) || 0, l._pt = {
            _next: l._pt,
            p: p || u === 1 ? p : ",",
            s: v,
            c: _.charAt(1) === "=" ? vr(v, _) - v : parseFloat(_) - v,
            m: f && f < 4 ? Math.round : 0
        }, c = hs.lastIndex);
        return l.c = c < i.length ? i.substring(c, i.length) : "", l.fp = a, (Oa.test(i) || T) && (l.e = 0), this._pt = l, l
    },
    uo = function(t, r, e, i, n, s, a, l, c, u) {
        dt(i) && (i = i(n || 0, t, s));
        var h = t[r],
            d = e !== "get" ? e : dt(h) ? c ? t[r.indexOf("set") || !dt(t["get" + r.substr(3)]) ? r : "get" + r.substr(3)](c) : t[r]() : h,
            f = dt(h) ? c ? du : ol : ho,
            _;
        if (Dt(i) && (~i.indexOf("random(") && (i = cn(i)), i.charAt(1) === "=" && (_ = vr(d, i) + (Vt(d) || 0), (_ || _ === 0) && (i = _))), !u || d !== i || Ys) return !isNaN(d * i) && i !== "" ? (_ = new le(this._pt, t, r, +d || 0, i - (d || 0), typeof h == "boolean" ? _u : al, 0, f), c && (_.fp = c), a && _.modifier(a, this, t), this._pt = _) : (!h && !(r in t) && oo(r, i), lu.call(this, t, r, d, i, f, l || xe.stringFilter, c))
    },
    cu = function(t, r, e, i, n) {
        if (dt(t) && (t = Kr(t, n, r, e, i)), !ti(t) || t.style && t.nodeType || Ut(t) || ka(t)) return Dt(t) ? Kr(t, n, r, e, i) : t;
        var s = {},
            a;
        for (a in t) s[a] = Kr(t[a], n, r, e, i);
        return s
    },
    rl = function(t, r, e, i, n, s) {
        var a, l, c, u;
        if (me[t] && (a = new me[t]).init(n, a.rawVars ? r[t] : cu(r[t], i, n, s, e), e, i, s) !== !1 && (e._pt = l = new le(e._pt, n, t, 0, 1, a.render, a, 0, a.priority), e !== _r))
            for (c = e._ptLookup[e._targets.indexOf(n)], u = a._props.length; u--;) c[a._props[u]] = l;
        return a
    },
    wi, Ys, fo = function o(t, r, e) {
        var i = t.vars,
            n = i.ease,
            s = i.startAt,
            a = i.immediateRender,
            l = i.lazy,
            c = i.onUpdate,
            u = i.runBackwards,
            h = i.yoyoEase,
            d = i.keyframes,
            f = i.autoRevert,
            _ = t._dur,
            p = t._startAt,
            v = t._targets,
            T = t.parent,
            b = T && T.data === "nested" ? T.vars.targets : v,
            E = t._overwrite === "auto" && !io,
            g = t.timeline,
            y, S, w, C, k, O, L, P, W, N, X, B, M;
        if (g && (!d || !n) && (n = "none"), t._ease = $i(n, kr.ease), t._yEase = h ? Ja($i(h === !0 ? n : h, kr.ease)) : 0, h && t._yoyo && !t._repeat && (h = t._yEase, t._yEase = t._ease, t._ease = h), t._from = !g && !!i.runBackwards, !g || d && !i.stagger) {
            if (P = v[0] ? Ui(v[0]).harness : 0, B = P && i[P.prop], y = qn(i, ao), p && (p._zTime < 0 && p.progress(1), r < 0 && u && a && !f ? p.render(-1, !0) : p.revert(u && _ ? Ln : zc), p._lazy = 0), s) {
                if (Ci(t._startAt = Tt.set(v, Re({
                        data: "isStart",
                        overwrite: !1,
                        parent: T,
                        immediateRender: !0,
                        lazy: !p && oe(l),
                        startAt: null,
                        delay: 0,
                        onUpdate: c && function() {
                            return be(t, "onUpdate")
                        },
                        stagger: 0
                    }, s))), t._startAt._dp = 0, t._startAt._sat = t, r < 0 && (Xt || !a && !f) && t._startAt.revert(Ln), a && _ && r <= 0 && e <= 0) {
                    r && (t._zTime = r);
                    return
                }
            } else if (u && _ && !p) {
                if (r && (a = !1), w = Re({
                        overwrite: !1,
                        data: "isFromStart",
                        lazy: a && !p && oe(l),
                        immediateRender: a,
                        stagger: 0,
                        parent: T
                    }, y), B && (w[P.prop] = B), Ci(t._startAt = Tt.set(v, w)), t._startAt._dp = 0, t._startAt._sat = t, r < 0 && (Xt ? t._startAt.revert(Ln) : t._startAt.render(-1, !0)), t._zTime = r, !a) o(t._startAt, rt, rt);
                else if (!r) return
            }
            for (t._pt = t._ptCache = 0, l = _ && oe(l) || l && !_, S = 0; S < v.length; S++) {
                if (k = v[S], L = k._gsap || co(v)[S]._gsap, t._ptLookup[S] = N = {}, Rs[L.id] && ki.length && Un(), X = b === v ? S : b.indexOf(k), P && (W = new P).init(k, B || y, t, X, b) !== !1 && (t._pt = C = new le(t._pt, k, W.name, 0, 1, W.render, W, 0, W.priority), W._props.forEach(function($) {
                        N[$] = C
                    }), W.priority && (O = 1)), !P || B)
                    for (w in y) me[w] && (W = rl(w, y, t, X, k, b)) ? W.priority && (O = 1) : N[w] = C = uo.call(t, k, w, "get", y[w], X, b, 0, i.stringFilter);
                t._op && t._op[S] && t.kill(k, t._op[S]), E && t._pt && (wi = t, ct.killTweensOf(k, N, t.globalTime(r)), M = !t.parent, wi = 0), t._pt && l && (Rs[L.id] = 1)
            }
            O && cl(t), t._onInit && t._onInit(t)
        }
        t._onUpdate = c, t._initted = (!t._op || t._pt) && !M, d && r <= 0 && g.render(Ae, !0, !0)
    },
    uu = function(t, r, e, i, n, s, a, l) {
        var c = (t._pt && t._ptCache || (t._ptCache = {}))[r],
            u, h, d, f;
        if (!c)
            for (c = t._ptCache[r] = [], d = t._ptLookup, f = t._targets.length; f--;) {
                if (u = d[f][r], u && u.d && u.d._pt)
                    for (u = u.d._pt; u && u.p !== r && u.fp !== r;) u = u._next;
                if (!u) return Ys = 1, t.vars[r] = "+=0", fo(t, a), Ys = 0, l ? an(r + " not eligible for reset") : 1;
                c.push(u)
            }
        for (f = c.length; f--;) h = c[f], u = h._pt || h, u.s = (i || i === 0) && !n ? i : u.s + (i || 0) + s * u.c, u.c = e - u.s, h.e && (h.e = mt(e) + Vt(h.e)), h.b && (h.b = u.s + Vt(h.b))
    },
    fu = function(t, r) {
        var e = t[0] ? Ui(t[0]).harness : 0,
            i = e && e.aliases,
            n, s, a, l;
        if (!i) return r;
        n = Qi({}, r);
        for (s in i)
            if (s in n)
                for (l = i[s].split(","), a = l.length; a--;) n[l[a]] = n[s];
        return n
    },
    hu = function(t, r, e, i) {
        var n = r.ease || i || "power1.inOut",
            s, a;
        if (Ut(r)) a = e[t] || (e[t] = []), r.forEach(function(l, c) {
            return a.push({
                t: c / (r.length - 1) * 100,
                v: l,
                e: n
            })
        });
        else
            for (s in r) a = e[s] || (e[s] = []), s === "ease" || a.push({
                t: parseFloat(t),
                v: r[s],
                e: n
            })
    },
    Kr = function(t, r, e, i, n) {
        return dt(t) ? t.call(r, e, i, n) : Dt(t) && ~t.indexOf("random(") ? cn(t) : t
    },
    nl = lo + "repeat,repeatDelay,yoyo,repeatRefresh,yoyoEase,autoRevert",
    sl = {};
ae(nl + ",id,stagger,delay,duration,paused,scrollTrigger", function(o) {
    return sl[o] = 1
});
var Tt = function(o) {
    Ta(t, o);

    function t(e, i, n, s) {
        var a;
        typeof i == "number" && (n.duration = i, i = n, n = null), a = o.call(this, s ? i : jr(i)) || this;
        var l = a.vars,
            c = l.duration,
            u = l.delay,
            h = l.immediateRender,
            d = l.stagger,
            f = l.overwrite,
            _ = l.keyframes,
            p = l.defaults,
            v = l.scrollTrigger,
            T = l.yoyoEase,
            b = i.parent || ct,
            E = (Ut(e) || ka(e) ? fi(e[0]) : "length" in i) ? [e] : Me(e),
            g, y, S, w, C, k, O, L;
        if (a._targets = E.length ? co(E) : an("GSAP target " + e + " not found. https://gsap.com", !xe.nullTargetWarn) || [], a._ptLookup = [], a._overwrite = f, _ || d || bn(c) || bn(u)) {
            if (i = a.vars, g = a.timeline = new ee({
                    data: "nested",
                    defaults: p || {},
                    targets: b && b.data === "nested" ? b.vars.targets : E
                }), g.kill(), g.parent = g._dp = si(a), g._start = 0, d || bn(c) || bn(u)) {
                if (w = E.length, O = d && Va(d), ti(d))
                    for (C in d) ~nl.indexOf(C) && (L || (L = {}), L[C] = d[C]);
                for (y = 0; y < w; y++) S = qn(i, sl), S.stagger = 0, T && (S.yoyoEase = T), L && Qi(S, L), k = E[y], S.duration = +Kr(c, si(a), y, k, E), S.delay = (+Kr(u, si(a), y, k, E) || 0) - a._delay, !d && w === 1 && S.delay && (a._delay = u = S.delay, a._start += u, S.delay = 0), g.to(k, S, O ? O(y, k, E) : 0), g._ease = K.none;
                g.duration() ? c = u = 0 : a.timeline = 0
            } else if (_) {
                jr(Re(g.vars.defaults, {
                    ease: "none"
                })), g._ease = $i(_.ease || i.ease || "none");
                var P = 0,
                    W, N, X;
                if (Ut(_)) _.forEach(function(B) {
                    return g.to(E, B, ">")
                }), g.duration();
                else {
                    S = {};
                    for (C in _) C === "ease" || C === "easeEach" || hu(C, _[C], S, _.easeEach);
                    for (C in S)
                        for (W = S[C].sort(function(B, M) {
                                return B.t - M.t
                            }), P = 0, y = 0; y < W.length; y++) N = W[y], X = {
                            ease: N.e,
                            duration: (N.t - (y ? W[y - 1].t : 0)) / 100 * c
                        }, X[C] = N.v, g.to(E, X, P), P += X.duration;
                    g.duration() < c && g.to({}, {
                        duration: c - g.duration()
                    })
                }
            }
            c || a.duration(c = g.duration())
        } else a.timeline = 0;
        return f === !0 && !io && (wi = si(a), ct.killTweensOf(E), wi = 0), Ge(b, si(a), n), i.reversed && a.reverse(), i.paused && a.paused(!0), (h || !c && !_ && a._start === Mt(b._time) && oe(h) && Vc(si(a)) && b.data !== "nested") && (a._tTime = -rt, a.render(Math.max(0, -u) || 0)), v && Ya(si(a), v), a
    }
    var r = t.prototype;
    return r.render = function(i, n, s) {
        var a = this._time,
            l = this._tDur,
            c = this._dur,
            u = i < 0,
            h = i > l - rt && !u ? l : i < rt ? 0 : i,
            d, f, _, p, v, T, b, E, g;
        if (!c) Uc(this, i, n, s);
        else if (h !== this._tTime || !i || s || !this._initted && this._tTime || this._startAt && this._zTime < 0 !== u) {
            if (d = h, E = this.timeline, this._repeat) {
                if (p = c + this._rDelay, this._repeat < -1 && u) return this.totalTime(p * 100 + i, n, s);
                if (d = Mt(h % p), h === l ? (_ = this._repeat, d = c) : (_ = ~~(h / p), _ && _ === Mt(h / p) && (d = c, _--), d > c && (d = c)), T = this._yoyo && _ & 1, T && (g = this._yEase, d = c - d), v = Er(this._tTime, p), d === a && !s && this._initted && _ === v) return this._tTime = h, this;
                _ !== v && (E && this._yEase && tl(E, T), this.vars.repeatRefresh && !T && !this._lock && this._time !== p && this._initted && (this._lock = s = 1, this.render(Mt(p * _), !0).invalidate()._lock = 0))
            }
            if (!this._initted) {
                if (Na(this, u ? i : d, s, n, h)) return this._tTime = 0, this;
                if (a !== this._time && !(s && this.vars.repeatRefresh && _ !== v)) return this;
                if (c !== this._dur) return this.render(i, n, s)
            }
            if (this._tTime = h, this._time = d, !this._act && this._ts && (this._act = 1, this._lazy = 0), this.ratio = b = (g || this._ease)(d / c), this._from && (this.ratio = b = 1 - b), d && !a && !n && !_ && (be(this, "onStart"), this._tTime !== h)) return this;
            for (f = this._pt; f;) f.r(b, f.d), f = f._next;
            E && E.render(i < 0 ? i : E._dur * E._ease(d / this._dur), n, s) || this._startAt && (this._zTime = i), this._onUpdate && !n && (u && Ls(this, i, n, s), be(this, "onUpdate")), this._repeat && _ !== v && this.vars.onRepeat && !n && this.parent && be(this, "onRepeat"), (h === this._tDur || !h) && this._tTime === h && (u && !this._onUpdate && Ls(this, i, !0, !0), (i || !c) && (h === this._tDur && this._ts > 0 || !h && this._ts < 0) && Ci(this, 1), !n && !(u && !a) && (h || a || T) && (be(this, h === l ? "onComplete" : "onReverseComplete", !0), this._prom && !(h < l && this.timeScale() > 0) && this._prom()))
        }
        return this
    }, r.targets = function() {
        return this._targets
    }, r.invalidate = function(i) {
        return (!i || !this.vars.runBackwards) && (this._startAt = 0), this._pt = this._op = this._onUpdate = this._lazy = this.ratio = 0, this._ptLookup = [], this.timeline && this.timeline.invalidate(i), o.prototype.invalidate.call(this, i)
    }, r.resetTo = function(i, n, s, a, l) {
        un || ye.wake(), this._ts || this.play();
        var c = Math.min(this._dur, (this._dp._time - this._start) * this._ts),
            u;
        return this._initted || fo(this, c), u = this._ease(c / this._dur), uu(this, i, n, s, a, u, c, l) ? this.resetTo(i, n, s, a, 1) : (os(this, 0), this.parent || za(this._dp, this, "_first", "_last", this._dp._sort ? "_start" : 0), this.render(0))
    }, r.kill = function(i, n) {
        if (n === void 0 && (n = "all"), !i && (!n || n === "all")) return this._lazy = this._pt = 0, this.parent ? Wr(this) : this;
        if (this.timeline) {
            var s = this.timeline.totalDuration();
            return this.timeline.killTweensOf(i, n, wi && wi.vars.overwrite !== !0)._first || Wr(this), this.parent && s !== this.timeline.totalDuration() && Or(this, this._dur * this.timeline._tDur / s, 0, 1), this
        }
        var a = this._targets,
            l = i ? Me(i) : a,
            c = this._ptLookup,
            u = this._pt,
            h, d, f, _, p, v, T;
        if ((!n || n === "all") && Wc(a, l)) return n === "all" && (this._pt = 0), Wr(this);
        for (h = this._op = this._op || [], n !== "all" && (Dt(n) && (p = {}, ae(n, function(b) {
                return p[b] = 1
            }), n = p), n = fu(a, n)), T = a.length; T--;)
            if (~l.indexOf(a[T])) {
                d = c[T], n === "all" ? (h[T] = n, _ = d, f = {}) : (f = h[T] = h[T] || {}, _ = n);
                for (p in _) v = d && d[p], v && ((!("kill" in v.d) || v.d.kill(p) === !0) && ns(this, v, "_pt"), delete d[p]), f !== "all" && (f[p] = 1)
            } return this._initted && !this._pt && u && Wr(this), this
    }, t.to = function(i, n) {
        return new t(i, n, arguments[2])
    }, t.from = function(i, n) {
        return Gr(1, arguments)
    }, t.delayedCall = function(i, n, s, a) {
        return new t(n, 0, {
            immediateRender: !1,
            lazy: !1,
            overwrite: !1,
            delay: i,
            onComplete: n,
            onReverseComplete: n,
            onCompleteParams: s,
            onReverseCompleteParams: s,
            callbackScope: a
        })
    }, t.fromTo = function(i, n, s) {
        return Gr(2, arguments)
    }, t.set = function(i, n) {
        return n.duration = 0, n.repeatDelay || (n.repeat = 0), new t(i, n)
    }, t.killTweensOf = function(i, n, s) {
        return ct.killTweensOf(i, n, s)
    }, t
}(fn);
Re(Tt.prototype, {
    _targets: [],
    _lazy: 0,
    _startAt: 0,
    _op: 0,
    _onInit: 0
});
ae("staggerTo,staggerFrom,staggerFromTo", function(o) {
    Tt[o] = function() {
        var t = new ee,
            r = Bs.call(arguments, 0);
        return r.splice(o === "staggerFromTo" ? 5 : 4, 0, 0), t[o].apply(t, r)
    }
});
var ho = function(t, r, e) {
        return t[r] = e
    },
    ol = function(t, r, e) {
        return t[r](e)
    },
    du = function(t, r, e, i) {
        return t[r](i.fp, e)
    },
    pu = function(t, r, e) {
        return t.setAttribute(r, e)
    },
    po = function(t, r) {
        return dt(t[r]) ? ol : ro(t[r]) && t.setAttribute ? pu : ho
    },
    al = function(t, r) {
        return r.set(r.t, r.p, Math.round((r.s + r.c * t) * 1e6) / 1e6, r)
    },
    _u = function(t, r) {
        return r.set(r.t, r.p, !!(r.s + r.c * t), r)
    },
    ll = function(t, r) {
        var e = r._pt,
            i = "";
        if (!t && r.b) i = r.b;
        else if (t === 1 && r.e) i = r.e;
        else {
            for (; e;) i = e.p + (e.m ? e.m(e.s + e.c * t) : Math.round((e.s + e.c * t) * 1e4) / 1e4) + i, e = e._next;
            i += r.c
        }
        r.set(r.t, r.p, i, r)
    },
    _o = function(t, r) {
        for (var e = r._pt; e;) e.r(t, e.d), e = e._next
    },
    gu = function(t, r, e, i) {
        for (var n = this._pt, s; n;) s = n._next, n.p === i && n.modifier(t, r, e), n = s
    },
    mu = function(t) {
        for (var r = this._pt, e, i; r;) i = r._next, r.p === t && !r.op || r.op === t ? ns(this, r, "_pt") : r.dep || (e = 1), r = i;
        return !e
    },
    vu = function(t, r, e, i) {
        i.mSet(t, r, i.m.call(i.tween, e, i.mt), i)
    },
    cl = function(t) {
        for (var r = t._pt, e, i, n, s; r;) {
            for (e = r._next, i = n; i && i.pr > r.pr;) i = i._next;
            (r._prev = i ? i._prev : s) ? r._prev._next = r: n = r, (r._next = i) ? i._prev = r : s = r, r = e
        }
        t._pt = n
    },
    le = function() {
        function o(r, e, i, n, s, a, l, c, u) {
            this.t = e, this.s = n, this.c = s, this.p = i, this.r = a || al, this.d = l || this, this.set = c || ho, this.pr = u || 0, this._next = r, r && (r._prev = this)
        }
        var t = o.prototype;
        return t.modifier = function(e, i, n) {
            this.mSet = this.mSet || this.set, this.set = vu, this.m = e, this.mt = n, this.tween = i
        }, o
    }();
ae(lo + "parent,duration,ease,delay,overwrite,runBackwards,startAt,yoyo,immediateRender,repeat,repeatDelay,data,paused,reversed,lazy,callbackScope,stringFilter,id,yoyoEase,stagger,inherit,repeatRefresh,keyframes,autoRevert,scrollTrigger", function(o) {
    return ao[o] = 1
});
Te.TweenMax = Te.TweenLite = Tt;
Te.TimelineLite = Te.TimelineMax = ee;
ct = new ee({
    sortChildren: !1,
    defaults: kr,
    autoRemoveChildren: !0,
    id: "root",
    smoothChildTiming: !0
});
xe.stringFilter = Qa;
var ji = [],
    Bn = {},
    yu = [],
    Bo = 0,
    wu = 0,
    ms = function(t) {
        return (Bn[t] || yu).map(function(r) {
            return r()
        })
    },
    Ns = function() {
        var t = Date.now(),
            r = [];
        t - Bo > 2 && (ms("matchMediaInit"), ji.forEach(function(e) {
            var i = e.queries,
                n = e.conditions,
                s, a, l, c;
            for (a in i) s = $e.matchMedia(i[a]).matches, s && (l = 1), s !== n[a] && (n[a] = s, c = 1);
            c && (e.revert(), l && r.push(e))
        }), ms("matchMediaRevert"), r.forEach(function(e) {
            return e.onMatch(e, function(i) {
                return e.add(null, i)
            })
        }), Bo = t, ms("matchMedia"))
    },
    ul = function() {
        function o(r, e) {
            this.selector = e && zs(e), this.data = [], this._r = [], this.isReverted = !1, this.id = wu++, r && this.add(r)
        }
        var t = o.prototype;
        return t.add = function(e, i, n) {
            dt(e) && (n = i, i = e, e = dt);
            var s = this,
                a = function() {
                    var c = ot,
                        u = s.selector,
                        h;
                    return c && c !== s && c.data.push(s), n && (s.selector = zs(n)), ot = s, h = i.apply(s, arguments), dt(h) && s._r.push(h), ot = c, s.selector = u, s.isReverted = !1, h
                };
            return s.last = a, e === dt ? a(s, function(l) {
                return s.add(null, l)
            }) : e ? s[e] = a : a
        }, t.ignore = function(e) {
            var i = ot;
            ot = null, e(this), ot = i
        }, t.getTweens = function() {
            var e = [];
            return this.data.forEach(function(i) {
                return i instanceof o ? e.push.apply(e, i.getTweens()) : i instanceof Tt && !(i.parent && i.parent.data === "nested") && e.push(i)
            }), e
        }, t.clear = function() {
            this._r.length = this.data.length = 0
        }, t.kill = function(e, i) {
            var n = this;
            if (e ? function() {
                    for (var a = n.getTweens(), l = n.data.length, c; l--;) c = n.data[l], c.data === "isFlip" && (c.revert(), c.getChildren(!0, !0, !1).forEach(function(u) {
                        return a.splice(a.indexOf(u), 1)
                    }));
                    for (a.map(function(u) {
                            return {
                                g: u._dur || u._delay || u._sat && !u._sat.vars.immediateRender ? u.globalTime(0) : -1 / 0,
                                t: u
                            }
                        }).sort(function(u, h) {
                            return h.g - u.g || -1 / 0
                        }).forEach(function(u) {
                            return u.t.revert(e)
                        }), l = n.data.length; l--;) c = n.data[l], c instanceof ee ? c.data !== "nested" && (c.scrollTrigger && c.scrollTrigger.revert(), c.kill()) : !(c instanceof Tt) && c.revert && c.revert(e);
                    n._r.forEach(function(u) {
                        return u(e, n)
                    }), n.isReverted = !0
                }() : this.data.forEach(function(a) {
                    return a.kill && a.kill()
                }), this.clear(), i)
                for (var s = ji.length; s--;) ji[s].id === this.id && ji.splice(s, 1)
        }, t.revert = function(e) {
            this.kill(e || {})
        }, o
    }(),
    bu = function() {
        function o(r) {
            this.contexts = [], this.scope = r, ot && ot.data.push(this)
        }
        var t = o.prototype;
        return t.add = function(e, i, n) {
            ti(e) || (e = {
                matches: e
            });
            var s = new ul(0, n || this.scope),
                a = s.conditions = {},
                l, c, u;
            ot && !s.selector && (s.selector = ot.selector), this.contexts.push(s), i = s.add("onMatch", i), s.queries = e;
            for (c in e) c === "all" ? u = 1 : (l = $e.matchMedia(e[c]), l && (ji.indexOf(s) < 0 && ji.push(s), (a[c] = l.matches) && (u = 1), l.addListener ? l.addListener(Ns) : l.addEventListener("change", Ns)));
            return u && i(s, function(h) {
                return s.add(null, h)
            }), this
        }, t.revert = function(e) {
            this.kill(e || {})
        }, t.kill = function(e) {
            this.contexts.forEach(function(i) {
                return i.kill(e, !0)
            })
        }, o
    }(),
    jn = {
        registerPlugin: function() {
            for (var t = arguments.length, r = new Array(t), e = 0; e < t; e++) r[e] = arguments[e];
            r.forEach(function(i) {
                return Ga(i)
            })
        },
        timeline: function(t) {
            return new ee(t)
        },
        getTweensOf: function(t, r) {
            return ct.getTweensOf(t, r)
        },
        getProperty: function(t, r, e, i) {
            Dt(t) && (t = Me(t)[0]);
            var n = Ui(t || {}).get,
                s = e ? Ba : Ia;
            return e === "native" && (e = ""), t && (r ? s((me[r] && me[r].get || n)(t, r, e, i)) : function(a, l, c) {
                return s((me[a] && me[a].get || n)(t, a, l, c))
            })
        },
        quickSetter: function(t, r, e) {
            if (t = Me(t), t.length > 1) {
                var i = t.map(function(u) {
                        return ue.quickSetter(u, r, e)
                    }),
                    n = i.length;
                return function(u) {
                    for (var h = n; h--;) i[h](u)
                }
            }
            t = t[0] || {};
            var s = me[r],
                a = Ui(t),
                l = a.harness && (a.harness.aliases || {})[r] || r,
                c = s ? function(u) {
                    var h = new s;
                    _r._pt = 0, h.init(t, e ? u + e : u, _r, 0, [t]), h.render(1, h), _r._pt && _o(1, _r)
                } : a.set(t, l);
            return s ? c : function(u) {
                return c(t, l, e ? u + e : u, a, 1)
            }
        },
        quickTo: function(t, r, e) {
            var i, n = ue.to(t, Qi((i = {}, i[r] = "+=0.1", i.paused = !0, i), e || {})),
                s = function(l, c, u) {
                    return n.resetTo(r, l, c, u)
                };
            return s.tween = n, s
        },
        isTweening: function(t) {
            return ct.getTweensOf(t, !0).length > 0
        },
        defaults: function(t) {
            return t && t.ease && (t.ease = $i(t.ease, kr.ease)), Mo(kr, t || {})
        },
        config: function(t) {
            return Mo(xe, t || {})
        },
        registerEffect: function(t) {
            var r = t.name,
                e = t.effect,
                i = t.plugins,
                n = t.defaults,
                s = t.extendTimeline;
            (i || "").split(",").forEach(function(a) {
                return a && !me[a] && !Te[a] && an(r + " effect requires " + a + " plugin.")
            }), ds[r] = function(a, l, c) {
                return e(Me(a), Re(l || {}, n), c)
            }, s && (ee.prototype[r] = function(a, l, c) {
                return this.add(ds[r](a, ti(l) ? l : (c = l) && {}, this), c)
            })
        },
        registerEase: function(t, r) {
            K[t] = $i(r)
        },
        parseEase: function(t, r) {
            return arguments.length ? $i(t, r) : K
        },
        getById: function(t) {
            return ct.getById(t)
        },
        exportRoot: function(t, r) {
            t === void 0 && (t = {});
            var e = new ee(t),
                i, n;
            for (e.smoothChildTiming = oe(t.smoothChildTiming), ct.remove(e), e._dp = 0, e._time = e._tTime = ct._time, i = ct._first; i;) n = i._next, (r || !(!i._dur && i instanceof Tt && i.vars.onComplete === i._targets[0])) && Ge(e, i, i._start - i._delay), i = n;
            return Ge(ct, e, 0), e
        },
        context: function(t, r) {
            return t ? new ul(t, r) : ot
        },
        matchMedia: function(t) {
            return new bu(t)
        },
        matchMediaRefresh: function() {
            return ji.forEach(function(t) {
                var r = t.conditions,
                    e, i;
                for (i in r) r[i] && (r[i] = !1, e = 1);
                e && t.revert()
            }) || Ns()
        },
        addEventListener: function(t, r) {
            var e = Bn[t] || (Bn[t] = []);
            ~e.indexOf(r) || e.push(r)
        },
        removeEventListener: function(t, r) {
            var e = Bn[t],
                i = e && e.indexOf(r);
            i >= 0 && e.splice(i, 1)
        },
        utils: {
            wrap: Jc,
            wrapYoyo: tu,
            distribute: Va,
            random: Ua,
            snap: Xa,
            normalize: Qc,
            getUnit: Vt,
            clamp: jc,
            splitColor: Ka,
            toArray: Me,
            selector: zs,
            mapRange: $a,
            pipe: Kc,
            unitize: Zc,
            interpolate: eu,
            shuffle: Ha
        },
        install: Aa,
        effects: ds,
        ticker: ye,
        updateRoot: ee.updateRoot,
        plugins: me,
        globalTimeline: ct,
        core: {
            PropTween: le,
            globals: Ma,
            Tween: Tt,
            Timeline: ee,
            Animation: fn,
            getCache: Ui,
            _removeLinkedListItem: ns,
            reverting: function() {
                return Xt
            },
            context: function(t) {
                return t && ot && (ot.data.push(t), t._ctx = ot), ot
            },
            suppressOverwrites: function(t) {
                return io = t
            }
        }
    };
ae("to,from,fromTo,delayedCall,set,killTweensOf", function(o) {
    return jn[o] = Tt[o]
});
ye.add(ee.updateRoot);
_r = jn.to({}, {
    duration: 0
});
var xu = function(t, r) {
        for (var e = t._pt; e && e.p !== r && e.op !== r && e.fp !== r;) e = e._next;
        return e
    },
    Tu = function(t, r) {
        var e = t._targets,
            i, n, s;
        for (i in r)
            for (n = e.length; n--;) s = t._ptLookup[n][i], s && (s = s.d) && (s._pt && (s = xu(s, i)), s && s.modifier && s.modifier(r[i], t, e[n], i))
    },
    vs = function(t, r) {
        return {
            name: t,
            rawVars: 1,
            init: function(i, n, s) {
                s._onInit = function(a) {
                    var l, c;
                    if (Dt(n) && (l = {}, ae(n, function(u) {
                            return l[u] = 1
                        }), n = l), r) {
                        l = {};
                        for (c in n) l[c] = r(n[c]);
                        n = l
                    }
                    Tu(a, n)
                }
            }
        }
    },
    ue = jn.registerPlugin({
        name: "attr",
        init: function(t, r, e, i, n) {
            var s, a, l;
            this.tween = e;
            for (s in r) l = t.getAttribute(s) || "", a = this.add(t, "setAttribute", (l || 0) + "", r[s], i, n, 0, 0, s), a.op = s, a.b = l, this._props.push(s)
        },
        render: function(t, r) {
            for (var e = r._pt; e;) Xt ? e.set(e.t, e.p, e.b, e) : e.r(t, e.d), e = e._next
        }
    }, {
        name: "endArray",
        init: function(t, r) {
            for (var e = r.length; e--;) this.add(t, e, t[e] || 0, r[e], 0, 0, 0, 0, 0, 1)
        }
    }, vs("roundProps", Fs), vs("modifiers"), vs("snap", Xa)) || jn;
Tt.version = ee.version = ue.version = "3.12.5";
Pa = 1;
no() && Cr();
K.Power0;
K.Power1;
K.Power2;
K.Power3;
K.Power4;
K.Linear;
K.Quad;
K.Cubic;
K.Quart;
K.Quint;
K.Strong;
K.Elastic;
K.Back;
K.SteppedEase;
K.Bounce;
K.Sine;
K.Expo;
K.Circ;
/*!
 * CSSPlugin 3.12.5
 * https://gsap.com
 *
 * Copyright 2008-2024, GreenSock. All rights reserved.
 * Subject to the terms at https://gsap.com/standard-license or for
 * Club GSAP members, the agreement issued with that membership.
 * @author: Jack Doyle, jack@greensock.com
 */
var zo, bi, yr, go, Hi, Fo, mo, Su = function() {
        return typeof window < "u"
    },
    hi = {},
    Yi = 180 / Math.PI,
    wr = Math.PI / 180,
    lr = Math.atan2,
    Yo = 1e8,
    vo = /([A-Z])/g,
    ku = /(left|right|width|margin|padding|x)/i,
    Eu = /[\s,\(]\S/,
    Ke = {
        autoAlpha: "opacity,visibility",
        scale: "scaleX,scaleY",
        alpha: "opacity"
    },
    Ws = function(t, r) {
        return r.set(r.t, r.p, Math.round((r.s + r.c * t) * 1e4) / 1e4 + r.u, r)
    },
    Ou = function(t, r) {
        return r.set(r.t, r.p, t === 1 ? r.e : Math.round((r.s + r.c * t) * 1e4) / 1e4 + r.u, r)
    },
    Cu = function(t, r) {
        return r.set(r.t, r.p, t ? Math.round((r.s + r.c * t) * 1e4) / 1e4 + r.u : r.b, r)
    },
    Pu = function(t, r) {
        var e = r.s + r.c * t;
        r.set(r.t, r.p, ~~(e + (e < 0 ? -.5 : .5)) + r.u, r)
    },
    fl = function(t, r) {
        return r.set(r.t, r.p, t ? r.e : r.b, r)
    },
    hl = function(t, r) {
        return r.set(r.t, r.p, t !== 1 ? r.b : r.e, r)
    },
    Au = function(t, r, e) {
        return t.style[r] = e
    },
    Mu = function(t, r, e) {
        return t.style.setProperty(r, e)
    },
    Du = function(t, r, e) {
        return t._gsap[r] = e
    },
    Ru = function(t, r, e) {
        return t._gsap.scaleX = t._gsap.scaleY = e
    },
    Lu = function(t, r, e, i, n) {
        var s = t._gsap;
        s.scaleX = s.scaleY = e, s.renderTransform(n, s)
    },
    Iu = function(t, r, e, i, n) {
        var s = t._gsap;
        s[r] = e, s.renderTransform(n, s)
    },
    ut = "transform",
    ce = ut + "Origin",
    Bu = function o(t, r) {
        var e = this,
            i = this.target,
            n = i.style,
            s = i._gsap;
        if (t in hi && n) {
            if (this.tfm = this.tfm || {}, t !== "transform") t = Ke[t] || t, ~t.indexOf(",") ? t.split(",").forEach(function(a) {
                return e.tfm[a] = oi(i, a)
            }) : this.tfm[t] = s.x ? s[t] : oi(i, t), t === ce && (this.tfm.zOrigin = s.zOrigin);
            else return Ke.transform.split(",").forEach(function(a) {
                return o.call(e, a, r)
            });
            if (this.props.indexOf(ut) >= 0) return;
            s.svg && (this.svgo = i.getAttribute("data-svg-origin"), this.props.push(ce, r, "")), t = ut
        }(n || r) && this.props.push(t, r, n[t])
    },
    dl = function(t) {
        t.translate && (t.removeProperty("translate"), t.removeProperty("scale"), t.removeProperty("rotate"))
    },
    zu = function() {
        var t = this.props,
            r = this.target,
            e = r.style,
            i = r._gsap,
            n, s;
        for (n = 0; n < t.length; n += 3) t[n + 1] ? r[t[n]] = t[n + 2] : t[n + 2] ? e[t[n]] = t[n + 2] : e.removeProperty(t[n].substr(0, 2) === "--" ? t[n] : t[n].replace(vo, "-$1").toLowerCase());
        if (this.tfm) {
            for (s in this.tfm) i[s] = this.tfm[s];
            i.svg && (i.renderTransform(), r.setAttribute("data-svg-origin", this.svgo || "")), n = mo(), (!n || !n.isStart) && !e[ut] && (dl(e), i.zOrigin && e[ce] && (e[ce] += " " + i.zOrigin + "px", i.zOrigin = 0, i.renderTransform()), i.uncache = 1)
        }
    },
    pl = function(t, r) {
        var e = {
            target: t,
            props: [],
            revert: zu,
            save: Bu
        };
        return t._gsap || ue.core.getCache(t), r && r.split(",").forEach(function(i) {
            return e.save(i)
        }), e
    },
    _l, Hs = function(t, r) {
        var e = bi.createElementNS ? bi.createElementNS((r || "http://www.w3.org/1999/xhtml").replace(/^https/, "http"), t) : bi.createElement(t);
        return e && e.style ? e : bi.createElement(t)
    },
    Qe = function o(t, r, e) {
        var i = getComputedStyle(t);
        return i[r] || i.getPropertyValue(r.replace(vo, "-$1").toLowerCase()) || i.getPropertyValue(r) || !e && o(t, Pr(r) || r, 1) || ""
    },
    No = "O,Moz,ms,Ms,Webkit".split(","),
    Pr = function(t, r, e) {
        var i = r || Hi,
            n = i.style,
            s = 5;
        if (t in n && !e) return t;
        for (t = t.charAt(0).toUpperCase() + t.substr(1); s-- && !(No[s] + t in n););
        return s < 0 ? null : (s === 3 ? "ms" : s >= 0 ? No[s] : "") + t
    },
    Vs = function() {
        Su() && window.document && (zo = window, bi = zo.document, yr = bi.documentElement, Hi = Hs("div") || {
            style: {}
        }, Hs("div"), ut = Pr(ut), ce = ut + "Origin", Hi.style.cssText = "border-width:0;line-height:0;position:absolute;padding:0", _l = !!Pr("perspective"), mo = ue.core.reverting, go = 1)
    },
    ys = function o(t) {
        var r = Hs("svg", this.ownerSVGElement && this.ownerSVGElement.getAttribute("xmlns") || "http://www.w3.org/2000/svg"),
            e = this.parentNode,
            i = this.nextSibling,
            n = this.style.cssText,
            s;
        if (yr.appendChild(r), r.appendChild(this), this.style.display = "block", t) try {
            s = this.getBBox(), this._gsapBBox = this.getBBox, this.getBBox = o
        } catch {} else this._gsapBBox && (s = this._gsapBBox());
        return e && (i ? e.insertBefore(this, i) : e.appendChild(this)), yr.removeChild(r), this.style.cssText = n, s
    },
    Wo = function(t, r) {
        for (var e = r.length; e--;)
            if (t.hasAttribute(r[e])) return t.getAttribute(r[e])
    },
    gl = function(t) {
        var r;
        try {
            r = t.getBBox()
        } catch {
            r = ys.call(t, !0)
        }
        return r && (r.width || r.height) || t.getBBox === ys || (r = ys.call(t, !0)), r && !r.width && !r.x && !r.y ? {
            x: +Wo(t, ["x", "cx", "x1"]) || 0,
            y: +Wo(t, ["y", "cy", "y1"]) || 0,
            width: 0,
            height: 0
        } : r
    },
    ml = function(t) {
        return !!(t.getCTM && (!t.parentNode || t.ownerSVGElement) && gl(t))
    },
    Ji = function(t, r) {
        if (r) {
            var e = t.style,
                i;
            r in hi && r !== ce && (r = ut), e.removeProperty ? (i = r.substr(0, 2), (i === "ms" || r.substr(0, 6) === "webkit") && (r = "-" + r), e.removeProperty(i === "--" ? r : r.replace(vo, "-$1").toLowerCase())) : e.removeAttribute(r)
        }
    },
    xi = function(t, r, e, i, n, s) {
        var a = new le(t._pt, r, e, 0, 1, s ? hl : fl);
        return t._pt = a, a.b = i, a.e = n, t._props.push(e), a
    },
    Ho = {
        deg: 1,
        rad: 1,
        turn: 1
    },
    Fu = {
        grid: 1,
        flex: 1
    },
    Pi = function o(t, r, e, i) {
        var n = parseFloat(e) || 0,
            s = (e + "").trim().substr((n + "").length) || "px",
            a = Hi.style,
            l = ku.test(r),
            c = t.tagName.toLowerCase() === "svg",
            u = (c ? "client" : "offset") + (l ? "Width" : "Height"),
            h = 100,
            d = i === "px",
            f = i === "%",
            _, p, v, T;
        if (i === s || !n || Ho[i] || Ho[s]) return n;
        if (s !== "px" && !d && (n = o(t, r, e, "px")), T = t.getCTM && ml(t), (f || s === "%") && (hi[r] || ~r.indexOf("adius"))) return _ = T ? t.getBBox()[l ? "width" : "height"] : t[u], mt(f ? n / _ * h : n / 100 * _);
        if (a[l ? "width" : "height"] = h + (d ? s : i), p = ~r.indexOf("adius") || i === "em" && t.appendChild && !c ? t : t.parentNode, T && (p = (t.ownerSVGElement || {}).parentNode), (!p || p === bi || !p.appendChild) && (p = bi.body), v = p._gsap, v && f && v.width && l && v.time === ye.time && !v.uncache) return mt(n / v.width * h);
        if (f && (r === "height" || r === "width")) {
            var b = t.style[r];
            t.style[r] = h + i, _ = t[u], b ? t.style[r] = b : Ji(t, r)
        } else(f || s === "%") && !Fu[Qe(p, "display")] && (a.position = Qe(t, "position")), p === t && (a.position = "static"), p.appendChild(Hi), _ = Hi[u], p.removeChild(Hi), a.position = "absolute";
        return l && f && (v = Ui(p), v.time = ye.time, v.width = p[u]), mt(d ? _ * n / h : _ && n ? h / _ * n : 0)
    },
    oi = function(t, r, e, i) {
        var n;
        return go || Vs(), r in Ke && r !== "transform" && (r = Ke[r], ~r.indexOf(",") && (r = r.split(",")[0])), hi[r] && r !== "transform" ? (n = dn(t, i), n = r !== "transformOrigin" ? n[r] : n.svg ? n.origin : Kn(Qe(t, ce)) + " " + n.zOrigin + "px") : (n = t.style[r], (!n || n === "auto" || i || ~(n + "").indexOf("calc(")) && (n = Gn[r] && Gn[r](t, r, e) || Qe(t, r) || Ra(t, r) || (r === "opacity" ? 1 : 0))), e && !~(n + "").trim().indexOf(" ") ? Pi(t, r, n, e) + e : n
    },
    Yu = function(t, r, e, i) {
        if (!e || e === "none") {
            var n = Pr(r, t, 1),
                s = n && Qe(t, n, 1);
            s && s !== e ? (r = n, e = s) : r === "borderColor" && (e = Qe(t, "borderTopColor"))
        }
        var a = new le(this._pt, t.style, r, 0, 1, ll),
            l = 0,
            c = 0,
            u, h, d, f, _, p, v, T, b, E, g, y;
        if (a.b = e, a.e = i, e += "", i += "", i === "auto" && (p = t.style[r], t.style[r] = i, i = Qe(t, r) || i, p ? t.style[r] = p : Ji(t, r)), u = [e, i], Qa(u), e = u[0], i = u[1], d = e.match(pr) || [], y = i.match(pr) || [], y.length) {
            for (; h = pr.exec(i);) v = h[0], b = i.substring(l, h.index), _ ? _ = (_ + 1) % 5 : (b.substr(-5) === "rgba(" || b.substr(-5) === "hsla(") && (_ = 1), v !== (p = d[c++] || "") && (f = parseFloat(p) || 0, g = p.substr((f + "").length), v.charAt(1) === "=" && (v = vr(f, v) + g), T = parseFloat(v), E = v.substr((T + "").length), l = pr.lastIndex - E.length, E || (E = E || xe.units[r] || g, l === i.length && (i += E, a.e += E)), g !== E && (f = Pi(t, r, p, E) || 0), a._pt = {
                _next: a._pt,
                p: b || c === 1 ? b : ",",
                s: f,
                c: T - f,
                m: _ && _ < 4 || r === "zIndex" ? Math.round : 0
            });
            a.c = l < i.length ? i.substring(l, i.length) : ""
        } else a.r = r === "display" && i === "none" ? hl : fl;
        return Oa.test(i) && (a.e = 0), this._pt = a, a
    },
    Vo = {
        top: "0%",
        bottom: "100%",
        left: "0%",
        right: "100%",
        center: "50%"
    },
    Nu = function(t) {
        var r = t.split(" "),
            e = r[0],
            i = r[1] || "50%";
        return (e === "top" || e === "bottom" || i === "left" || i === "right") && (t = e, e = i, i = t), r[0] = Vo[e] || e, r[1] = Vo[i] || i, r.join(" ")
    },
    Wu = function(t, r) {
        if (r.tween && r.tween._time === r.tween._dur) {
            var e = r.t,
                i = e.style,
                n = r.u,
                s = e._gsap,
                a, l, c;
            if (n === "all" || n === !0) i.cssText = "", l = 1;
            else
                for (n = n.split(","), c = n.length; --c > -1;) a = n[c], hi[a] && (l = 1, a = a === "transformOrigin" ? ce : ut), Ji(e, a);
            l && (Ji(e, ut), s && (s.svg && e.removeAttribute("transform"), dn(e, 1), s.uncache = 1, dl(i)))
        }
    },
    Gn = {
        clearProps: function(t, r, e, i, n) {
            if (n.data !== "isFromStart") {
                var s = t._pt = new le(t._pt, r, e, 0, 0, Wu);
                return s.u = i, s.pr = -10, s.tween = n, t._props.push(e), 1
            }
        }
    },
    hn = [1, 0, 0, 1, 0, 0],
    vl = {},
    yl = function(t) {
        return t === "matrix(1, 0, 0, 1, 0, 0)" || t === "none" || !t
    },
    Xo = function(t) {
        var r = Qe(t, ut);
        return yl(r) ? hn : r.substr(7).match(Ea).map(mt)
    },
    yo = function(t, r) {
        var e = t._gsap || Ui(t),
            i = t.style,
            n = Xo(t),
            s, a, l, c;
        return e.svg && t.getAttribute("transform") ? (l = t.transform.baseVal.consolidate().matrix, n = [l.a, l.b, l.c, l.d, l.e, l.f], n.join(",") === "1,0,0,1,0,0" ? hn : n) : (n === hn && !t.offsetParent && t !== yr && !e.svg && (l = i.display, i.display = "block", s = t.parentNode, (!s || !t.offsetParent) && (c = 1, a = t.nextElementSibling, yr.appendChild(t)), n = Xo(t), l ? i.display = l : Ji(t, "display"), c && (a ? s.insertBefore(t, a) : s ? s.appendChild(t) : yr.removeChild(t))), r && n.length > 6 ? [n[0], n[1], n[4], n[5], n[12], n[13]] : n)
    },
    Xs = function(t, r, e, i, n, s) {
        var a = t._gsap,
            l = n || yo(t, !0),
            c = a.xOrigin || 0,
            u = a.yOrigin || 0,
            h = a.xOffset || 0,
            d = a.yOffset || 0,
            f = l[0],
            _ = l[1],
            p = l[2],
            v = l[3],
            T = l[4],
            b = l[5],
            E = r.split(" "),
            g = parseFloat(E[0]) || 0,
            y = parseFloat(E[1]) || 0,
            S, w, C, k;
        e ? l !== hn && (w = f * v - _ * p) && (C = g * (v / w) + y * (-p / w) + (p * b - v * T) / w, k = g * (-_ / w) + y * (f / w) - (f * b - _ * T) / w, g = C, y = k) : (S = gl(t), g = S.x + (~E[0].indexOf("%") ? g / 100 * S.width : g), y = S.y + (~(E[1] || E[0]).indexOf("%") ? y / 100 * S.height : y)), i || i !== !1 && a.smooth ? (T = g - c, b = y - u, a.xOffset = h + (T * f + b * p) - T, a.yOffset = d + (T * _ + b * v) - b) : a.xOffset = a.yOffset = 0, a.xOrigin = g, a.yOrigin = y, a.smooth = !!i, a.origin = r, a.originIsAbsolute = !!e, t.style[ce] = "0px 0px", s && (xi(s, a, "xOrigin", c, g), xi(s, a, "yOrigin", u, y), xi(s, a, "xOffset", h, a.xOffset), xi(s, a, "yOffset", d, a.yOffset)), t.setAttribute("data-svg-origin", g + " " + y)
    },
    dn = function(t, r) {
        var e = t._gsap || new il(t);
        if ("x" in e && !r && !e.uncache) return e;
        var i = t.style,
            n = e.scaleX < 0,
            s = "px",
            a = "deg",
            l = getComputedStyle(t),
            c = Qe(t, ce) || "0",
            u, h, d, f, _, p, v, T, b, E, g, y, S, w, C, k, O, L, P, W, N, X, B, M, $, J, m, tt, qt, Le, ft, Rt;
        return u = h = d = p = v = T = b = E = g = 0, f = _ = 1, e.svg = !!(t.getCTM && ml(t)), l.translate && ((l.translate !== "none" || l.scale !== "none" || l.rotate !== "none") && (i[ut] = (l.translate !== "none" ? "translate3d(" + (l.translate + " 0 0").split(" ").slice(0, 3).join(", ") + ") " : "") + (l.rotate !== "none" ? "rotate(" + l.rotate + ") " : "") + (l.scale !== "none" ? "scale(" + l.scale.split(" ").join(",") + ") " : "") + (l[ut] !== "none" ? l[ut] : "")), i.scale = i.rotate = i.translate = "none"), w = yo(t, e.svg), e.svg && (e.uncache ? ($ = t.getBBox(), c = e.xOrigin - $.x + "px " + (e.yOrigin - $.y) + "px", M = "") : M = !r && t.getAttribute("data-svg-origin"), Xs(t, M || c, !!M || e.originIsAbsolute, e.smooth !== !1, w)), y = e.xOrigin || 0, S = e.yOrigin || 0, w !== hn && (L = w[0], P = w[1], W = w[2], N = w[3], u = X = w[4], h = B = w[5], w.length === 6 ? (f = Math.sqrt(L * L + P * P), _ = Math.sqrt(N * N + W * W), p = L || P ? lr(P, L) * Yi : 0, b = W || N ? lr(W, N) * Yi + p : 0, b && (_ *= Math.abs(Math.cos(b * wr))), e.svg && (u -= y - (y * L + S * W), h -= S - (y * P + S * N))) : (Rt = w[6], Le = w[7], m = w[8], tt = w[9], qt = w[10], ft = w[11], u = w[12], h = w[13], d = w[14], C = lr(Rt, qt), v = C * Yi, C && (k = Math.cos(-C), O = Math.sin(-C), M = X * k + m * O, $ = B * k + tt * O, J = Rt * k + qt * O, m = X * -O + m * k, tt = B * -O + tt * k, qt = Rt * -O + qt * k, ft = Le * -O + ft * k, X = M, B = $, Rt = J), C = lr(-W, qt), T = C * Yi, C && (k = Math.cos(-C), O = Math.sin(-C), M = L * k - m * O, $ = P * k - tt * O, J = W * k - qt * O, ft = N * O + ft * k, L = M, P = $, W = J), C = lr(P, L), p = C * Yi, C && (k = Math.cos(C), O = Math.sin(C), M = L * k + P * O, $ = X * k + B * O, P = P * k - L * O, B = B * k - X * O, L = M, X = $), v && Math.abs(v) + Math.abs(p) > 359.9 && (v = p = 0, T = 180 - T), f = mt(Math.sqrt(L * L + P * P + W * W)), _ = mt(Math.sqrt(B * B + Rt * Rt)), C = lr(X, B), b = Math.abs(C) > 2e-4 ? C * Yi : 0, g = ft ? 1 / (ft < 0 ? -ft : ft) : 0), e.svg && (M = t.getAttribute("transform"), e.forceCSS = t.setAttribute("transform", "") || !yl(Qe(t, ut)), M && t.setAttribute("transform", M))), Math.abs(b) > 90 && Math.abs(b) < 270 && (n ? (f *= -1, b += p <= 0 ? 180 : -180, p += p <= 0 ? 180 : -180) : (_ *= -1, b += b <= 0 ? 180 : -180)), r = r || e.uncache, e.x = u - ((e.xPercent = u && (!r && e.xPercent || (Math.round(t.offsetWidth / 2) === Math.round(-u) ? -50 : 0))) ? t.offsetWidth * e.xPercent / 100 : 0) + s, e.y = h - ((e.yPercent = h && (!r && e.yPercent || (Math.round(t.offsetHeight / 2) === Math.round(-h) ? -50 : 0))) ? t.offsetHeight * e.yPercent / 100 : 0) + s, e.z = d + s, e.scaleX = mt(f), e.scaleY = mt(_), e.rotation = mt(p) + a, e.rotationX = mt(v) + a, e.rotationY = mt(T) + a, e.skewX = b + a, e.skewY = E + a, e.transformPerspective = g + s, (e.zOrigin = parseFloat(c.split(" ")[2]) || !r && e.zOrigin || 0) && (i[ce] = Kn(c)), e.xOffset = e.yOffset = 0, e.force3D = xe.force3D, e.renderTransform = e.svg ? Vu : _l ? wl : Hu, e.uncache = 0, e
    },
    Kn = function(t) {
        return (t = t.split(" "))[0] + " " + t[1]
    },
    ws = function(t, r, e) {
        var i = Vt(r);
        return mt(parseFloat(r) + parseFloat(Pi(t, "x", e + "px", i))) + i
    },
    Hu = function(t, r) {
        r.z = "0px", r.rotationY = r.rotationX = "0deg", r.force3D = 0, wl(t, r)
    },
    zi = "0deg",
    zr = "0px",
    Fi = ") ",
    wl = function(t, r) {
        var e = r || this,
            i = e.xPercent,
            n = e.yPercent,
            s = e.x,
            a = e.y,
            l = e.z,
            c = e.rotation,
            u = e.rotationY,
            h = e.rotationX,
            d = e.skewX,
            f = e.skewY,
            _ = e.scaleX,
            p = e.scaleY,
            v = e.transformPerspective,
            T = e.force3D,
            b = e.target,
            E = e.zOrigin,
            g = "",
            y = T === "auto" && t && t !== 1 || T === !0;
        if (E && (h !== zi || u !== zi)) {
            var S = parseFloat(u) * wr,
                w = Math.sin(S),
                C = Math.cos(S),
                k;
            S = parseFloat(h) * wr, k = Math.cos(S), s = ws(b, s, w * k * -E), a = ws(b, a, -Math.sin(S) * -E), l = ws(b, l, C * k * -E + E)
        }
        v !== zr && (g += "perspective(" + v + Fi), (i || n) && (g += "translate(" + i + "%, " + n + "%) "), (y || s !== zr || a !== zr || l !== zr) && (g += l !== zr || y ? "translate3d(" + s + ", " + a + ", " + l + ") " : "translate(" + s + ", " + a + Fi), c !== zi && (g += "rotate(" + c + Fi), u !== zi && (g += "rotateY(" + u + Fi), h !== zi && (g += "rotateX(" + h + Fi), (d !== zi || f !== zi) && (g += "skew(" + d + ", " + f + Fi), (_ !== 1 || p !== 1) && (g += "scale(" + _ + ", " + p + Fi), b.style[ut] = g || "translate(0, 0)"
    },
    Vu = function(t, r) {
        var e = r || this,
            i = e.xPercent,
            n = e.yPercent,
            s = e.x,
            a = e.y,
            l = e.rotation,
            c = e.skewX,
            u = e.skewY,
            h = e.scaleX,
            d = e.scaleY,
            f = e.target,
            _ = e.xOrigin,
            p = e.yOrigin,
            v = e.xOffset,
            T = e.yOffset,
            b = e.forceCSS,
            E = parseFloat(s),
            g = parseFloat(a),
            y, S, w, C, k;
        l = parseFloat(l), c = parseFloat(c), u = parseFloat(u), u && (u = parseFloat(u), c += u, l += u), l || c ? (l *= wr, c *= wr, y = Math.cos(l) * h, S = Math.sin(l) * h, w = Math.sin(l - c) * -d, C = Math.cos(l - c) * d, c && (u *= wr, k = Math.tan(c - u), k = Math.sqrt(1 + k * k), w *= k, C *= k, u && (k = Math.tan(u), k = Math.sqrt(1 + k * k), y *= k, S *= k)), y = mt(y), S = mt(S), w = mt(w), C = mt(C)) : (y = h, C = d, S = w = 0), (E && !~(s + "").indexOf("px") || g && !~(a + "").indexOf("px")) && (E = Pi(f, "x", s, "px"), g = Pi(f, "y", a, "px")), (_ || p || v || T) && (E = mt(E + _ - (_ * y + p * w) + v), g = mt(g + p - (_ * S + p * C) + T)), (i || n) && (k = f.getBBox(), E = mt(E + i / 100 * k.width), g = mt(g + n / 100 * k.height)), k = "matrix(" + y + "," + S + "," + w + "," + C + "," + E + "," + g + ")", f.setAttribute("transform", k), b && (f.style[ut] = k)
    },
    Xu = function(t, r, e, i, n) {
        var s = 360,
            a = Dt(n),
            l = parseFloat(n) * (a && ~n.indexOf("rad") ? Yi : 1),
            c = l - i,
            u = i + c + "deg",
            h, d;
        return a && (h = n.split("_")[1], h === "short" && (c %= s, c !== c % (s / 2) && (c += c < 0 ? s : -s)), h === "cw" && c < 0 ? c = (c + s * Yo) % s - ~~(c / s) * s : h === "ccw" && c > 0 && (c = (c - s * Yo) % s - ~~(c / s) * s)), t._pt = d = new le(t._pt, r, e, i, c, Ou), d.e = u, d.u = "deg", t._props.push(e), d
    },
    Uo = function(t, r) {
        for (var e in r) t[e] = r[e];
        return t
    },
    Uu = function(t, r, e) {
        var i = Uo({}, e._gsap),
            n = "perspective,force3D,transformOrigin,svgOrigin",
            s = e.style,
            a, l, c, u, h, d, f, _;
        i.svg ? (c = e.getAttribute("transform"), e.setAttribute("transform", ""), s[ut] = r, a = dn(e, 1), Ji(e, ut), e.setAttribute("transform", c)) : (c = getComputedStyle(e)[ut], s[ut] = r, a = dn(e, 1), s[ut] = c);
        for (l in hi) c = i[l], u = a[l], c !== u && n.indexOf(l) < 0 && (f = Vt(c), _ = Vt(u), h = f !== _ ? Pi(e, l, c, _) : parseFloat(c), d = parseFloat(u), t._pt = new le(t._pt, a, l, h, d - h, Ws), t._pt.u = _ || 0, t._props.push(l));
        Uo(a, i)
    };
ae("padding,margin,Width,Radius", function(o, t) {
    var r = "Top",
        e = "Right",
        i = "Bottom",
        n = "Left",
        s = (t < 3 ? [r, e, i, n] : [r + n, r + e, i + e, i + n]).map(function(a) {
            return t < 2 ? o + a : "border" + a + o
        });
    Gn[t > 1 ? "border" + o : o] = function(a, l, c, u, h) {
        var d, f;
        if (arguments.length < 4) return d = s.map(function(_) {
            return oi(a, _, c)
        }), f = d.join(" "), f.split(d[0]).length === 5 ? d[0] : f;
        d = (u + "").split(" "), f = {}, s.forEach(function(_, p) {
            return f[_] = d[p] = d[p] || d[(p - 1) / 2 | 0]
        }), a.init(l, f, h)
    }
});
var bl = {
    name: "css",
    register: Vs,
    targetTest: function(t) {
        return t.style && t.nodeType
    },
    init: function(t, r, e, i, n) {
        var s = this._props,
            a = t.style,
            l = e.vars.startAt,
            c, u, h, d, f, _, p, v, T, b, E, g, y, S, w, C;
        go || Vs(), this.styles = this.styles || pl(t), C = this.styles.props, this.tween = e;
        for (p in r)
            if (p !== "autoRound" && (u = r[p], !(me[p] && rl(p, r, e, i, t, n)))) {
                if (f = typeof u, _ = Gn[p], f === "function" && (u = u.call(e, i, t, n), f = typeof u), f === "string" && ~u.indexOf("random(") && (u = cn(u)), _) _(this, t, p, u, e) && (w = 1);
                else if (p.substr(0, 2) === "--") c = (getComputedStyle(t).getPropertyValue(p) + "").trim(), u += "", Ei.lastIndex = 0, Ei.test(c) || (v = Vt(c), T = Vt(u)), T ? v !== T && (c = Pi(t, p, c, T) + T) : v && (u += v), this.add(a, "setProperty", c, u, i, n, 0, 0, p), s.push(p), C.push(p, 0, a[p]);
                else if (f !== "undefined") {
                    if (l && p in l ? (c = typeof l[p] == "function" ? l[p].call(e, i, t, n) : l[p], Dt(c) && ~c.indexOf("random(") && (c = cn(c)), Vt(c + "") || c === "auto" || (c += xe.units[p] || Vt(oi(t, p)) || ""), (c + "").charAt(1) === "=" && (c = oi(t, p))) : c = oi(t, p), d = parseFloat(c), b = f === "string" && u.charAt(1) === "=" && u.substr(0, 2), b && (u = u.substr(2)), h = parseFloat(u), p in Ke && (p === "autoAlpha" && (d === 1 && oi(t, "visibility") === "hidden" && h && (d = 0), C.push("visibility", 0, a.visibility), xi(this, a, "visibility", d ? "inherit" : "hidden", h ? "inherit" : "hidden", !h)), p !== "scale" && p !== "transform" && (p = Ke[p], ~p.indexOf(",") && (p = p.split(",")[0]))), E = p in hi, E) {
                        if (this.styles.save(p), g || (y = t._gsap, y.renderTransform && !r.parseTransform || dn(t, r.parseTransform), S = r.smoothOrigin !== !1 && y.smooth, g = this._pt = new le(this._pt, a, ut, 0, 1, y.renderTransform, y, 0, -1), g.dep = 1), p === "scale") this._pt = new le(this._pt, y, "scaleY", y.scaleY, (b ? vr(y.scaleY, b + h) : h) - y.scaleY || 0, Ws), this._pt.u = 0, s.push("scaleY", p), p += "X";
                        else if (p === "transformOrigin") {
                            C.push(ce, 0, a[ce]), u = Nu(u), y.svg ? Xs(t, u, 0, S, 0, this) : (T = parseFloat(u.split(" ")[2]) || 0, T !== y.zOrigin && xi(this, y, "zOrigin", y.zOrigin, T), xi(this, a, p, Kn(c), Kn(u)));
                            continue
                        } else if (p === "svgOrigin") {
                            Xs(t, u, 1, S, 0, this);
                            continue
                        } else if (p in vl) {
                            Xu(this, y, p, d, b ? vr(d, b + u) : u);
                            continue
                        } else if (p === "smoothOrigin") {
                            xi(this, y, "smooth", y.smooth, u);
                            continue
                        } else if (p === "force3D") {
                            y[p] = u;
                            continue
                        } else if (p === "transform") {
                            Uu(this, u, t);
                            continue
                        }
                    } else p in a || (p = Pr(p) || p);
                    if (E || (h || h === 0) && (d || d === 0) && !Eu.test(u) && p in a) v = (c + "").substr((d + "").length), h || (h = 0), T = Vt(u) || (p in xe.units ? xe.units[p] : v), v !== T && (d = Pi(t, p, c, T)), this._pt = new le(this._pt, E ? y : a, p, d, (b ? vr(d, b + h) : h) - d, !E && (T === "px" || p === "zIndex") && r.autoRound !== !1 ? Pu : Ws), this._pt.u = T || 0, v !== T && T !== "%" && (this._pt.b = c, this._pt.r = Cu);
                    else if (p in a) Yu.call(this, t, p, c, b ? b + u : u);
                    else if (p in t) this.add(t, p, c || t[p], b ? b + u : u, i, n);
                    else if (p !== "parseTransform") {
                        oo(p, u);
                        continue
                    }
                    E || (p in a ? C.push(p, 0, a[p]) : C.push(p, 1, c || t[p])), s.push(p)
                }
            } w && cl(this)
    },
    render: function(t, r) {
        if (r.tween._time || !mo())
            for (var e = r._pt; e;) e.r(t, e.d), e = e._next;
        else r.styles.revert()
    },
    get: oi,
    aliases: Ke,
    getSetter: function(t, r, e) {
        var i = Ke[r];
        return i && i.indexOf(",") < 0 && (r = i), r in hi && r !== ce && (t._gsap.x || oi(t, "x")) ? e && Fo === e ? r === "scale" ? Ru : Du : (Fo = e || {}) && (r === "scale" ? Lu : Iu) : t.style && !ro(t.style[r]) ? Au : ~r.indexOf("-") ? Mu : po(t, r)
    },
    core: {
        _removeProperty: Ji,
        _getMatrix: yo
    }
};
ue.utils.checkPrefix = Pr;
ue.core.getStyleSaver = pl;
(function(o, t, r, e) {
    var i = ae(o + "," + t + "," + r, function(n) {
        hi[n] = 1
    });
    ae(t, function(n) {
        xe.units[n] = "deg", vl[n] = 1
    }), Ke[i[13]] = o + "," + t, ae(e, function(n) {
        var s = n.split(":");
        Ke[s[1]] = i[s[0]]
    })
})("x,y,z,scale,scaleX,scaleY,xPercent,yPercent", "rotation,rotationX,rotationY,skewX,skewY", "transform,transformOrigin,svgOrigin,force3D,smoothOrigin,transformPerspective", "0:translateX,1:translateY,2:translateZ,8:rotate,8:rotationZ,8:rotateZ,9:rotateX,10:rotateY");
ae("x,y,z,top,right,bottom,left,width,height,fontSize,padding,margin,perspective", function(o) {
    xe.units[o] = "px"
});
ue.registerPlugin(bl);
var pe = ue.registerPlugin(bl) || ue;
pe.core.Tween;

function qu(o, t) {
    for (var r = 0; r < t.length; r++) {
        var e = t[r];
        e.enumerable = e.enumerable || !1, e.configurable = !0, "value" in e && (e.writable = !0), Object.defineProperty(o, e.key, e)
    }
}

function $u(o, t, r) {
    return t && qu(o.prototype, t), o
}
/*!
 * Observer 3.12.5
 * https://gsap.com
 *
 * @license Copyright 2008-2024, GreenSock. All rights reserved.
 * Subject to the terms at https://gsap.com/standard-license or for
 * Club GSAP members, the agreement issued with that membership.
 * @author: Jack Doyle, jack@greensock.com
 */
var Bt, zn, we, Ti, Si, br, xl, Ni, Zr, Tl, li, We, Sl, kl = function() {
        return Bt || typeof window < "u" && (Bt = window.gsap) && Bt.registerPlugin && Bt
    },
    El = 1,
    gr = [],
    q = [],
    Je = [],
    Qr = Date.now,
    Us = function(t, r) {
        return r
    },
    ju = function() {
        var t = Zr.core,
            r = t.bridge || {},
            e = t._scrollers,
            i = t._proxies;
        e.push.apply(e, q), i.push.apply(i, Je), q = e, Je = i, Us = function(s, a) {
            return r[s](a)
        }
    },
    Oi = function(t, r) {
        return ~Je.indexOf(t) && Je[Je.indexOf(t) + 1][r]
    },
    Jr = function(t) {
        return !!~Tl.indexOf(t)
    },
    Zt = function(t, r, e, i, n) {
        return t.addEventListener(r, e, {
            passive: i !== !1,
            capture: !!n
        })
    },
    Kt = function(t, r, e, i) {
        return t.removeEventListener(r, e, !!i)
    },
    xn = "scrollLeft",
    Tn = "scrollTop",
    qs = function() {
        return li && li.isPressed || q.cache++
    },
    Zn = function(t, r) {
        var e = function i(n) {
            if (n || n === 0) {
                El && (we.history.scrollRestoration = "manual");
                var s = li && li.isPressed;
                n = i.v = Math.round(n) || (li && li.iOS ? 1 : 0), t(n), i.cacheID = q.cache, s && Us("ss", n)
            } else(r || q.cache !== i.cacheID || Us("ref")) && (i.cacheID = q.cache, i.v = t());
            return i.v + i.offset
        };
        return e.offset = 0, t && e
    },
    ie = {
        s: xn,
        p: "left",
        p2: "Left",
        os: "right",
        os2: "Right",
        d: "width",
        d2: "Width",
        a: "x",
        sc: Zn(function(o) {
            return arguments.length ? we.scrollTo(o, Et.sc()) : we.pageXOffset || Ti[xn] || Si[xn] || br[xn] || 0
        })
    },
    Et = {
        s: Tn,
        p: "top",
        p2: "Top",
        os: "bottom",
        os2: "Bottom",
        d: "height",
        d2: "Height",
        a: "y",
        op: ie,
        sc: Zn(function(o) {
            return arguments.length ? we.scrollTo(ie.sc(), o) : we.pageYOffset || Ti[Tn] || Si[Tn] || br[Tn] || 0
        })
    },
    se = function(t, r) {
        return (r && r._ctx && r._ctx.selector || Bt.utils.toArray)(t)[0] || (typeof t == "string" && Bt.config().nullTargetWarn !== !1 ? console.warn("Element not found:", t) : null)
    },
    Ai = function(t, r) {
        var e = r.s,
            i = r.sc;
        Jr(t) && (t = Ti.scrollingElement || Si);
        var n = q.indexOf(t),
            s = i === Et.sc ? 1 : 2;
        !~n && (n = q.push(t) - 1), q[n + s] || Zt(t, "scroll", qs);
        var a = q[n + s],
            l = a || (q[n + s] = Zn(Oi(t, e), !0) || (Jr(t) ? i : Zn(function(c) {
                return arguments.length ? t[e] = c : t[e]
            })));
        return l.target = t, a || (l.smooth = Bt.getProperty(t, "scrollBehavior") === "smooth"), l
    },
    $s = function(t, r, e) {
        var i = t,
            n = t,
            s = Qr(),
            a = s,
            l = r || 50,
            c = Math.max(500, l * 3),
            u = function(_, p) {
                var v = Qr();
                p || v - s > l ? (n = i, i = _, a = s, s = v) : e ? i += _ : i = n + (_ - n) / (v - a) * (s - a)
            },
            h = function() {
                n = i = e ? 0 : i, a = s = 0
            },
            d = function(_) {
                var p = a,
                    v = n,
                    T = Qr();
                return (_ || _ === 0) && _ !== i && u(_), s === a || T - a > c ? 0 : (i + (e ? v : -v)) / ((e ? T : s) - p) * 1e3
            };
        return {
            update: u,
            reset: h,
            getVelocity: d
        }
    },
    Fr = function(t, r) {
        return r && !t._gsapAllow && t.preventDefault(), t.changedTouches ? t.changedTouches[0] : t
    },
    qo = function(t) {
        var r = Math.max.apply(Math, t),
            e = Math.min.apply(Math, t);
        return Math.abs(r) >= Math.abs(e) ? r : e
    },
    Ol = function() {
        Zr = Bt.core.globals().ScrollTrigger, Zr && Zr.core && ju()
    },
    Cl = function(t) {
        return Bt = t || kl(), !zn && Bt && typeof document < "u" && document.body && (we = window, Ti = document, Si = Ti.documentElement, br = Ti.body, Tl = [we, Ti, Si, br], Bt.utils.clamp, Sl = Bt.core.context || function() {}, Ni = "onpointerenter" in br ? "pointer" : "mouse", xl = vt.isTouch = we.matchMedia && we.matchMedia("(hover: none), (pointer: coarse)").matches ? 1 : "ontouchstart" in we || navigator.maxTouchPoints > 0 || navigator.msMaxTouchPoints > 0 ? 2 : 0, We = vt.eventTypes = ("ontouchstart" in Si ? "touchstart,touchmove,touchcancel,touchend" : "onpointerdown" in Si ? "pointerdown,pointermove,pointercancel,pointerup" : "mousedown,mousemove,mouseup,mouseup").split(","), setTimeout(function() {
            return El = 0
        }, 500), Ol(), zn = 1), zn
    };
ie.op = Et;
q.cache = 0;
var vt = function() {
    function o(r) {
        this.init(r)
    }
    var t = o.prototype;
    return t.init = function(e) {
        zn || Cl(Bt) || console.warn("Please gsap.registerPlugin(Observer)"), Zr || Ol();
        var i = e.tolerance,
            n = e.dragMinimum,
            s = e.type,
            a = e.target,
            l = e.lineHeight,
            c = e.debounce,
            u = e.preventDefault,
            h = e.onStop,
            d = e.onStopDelay,
            f = e.ignore,
            _ = e.wheelSpeed,
            p = e.event,
            v = e.onDragStart,
            T = e.onDragEnd,
            b = e.onDrag,
            E = e.onPress,
            g = e.onRelease,
            y = e.onRight,
            S = e.onLeft,
            w = e.onUp,
            C = e.onDown,
            k = e.onChangeX,
            O = e.onChangeY,
            L = e.onChange,
            P = e.onToggleX,
            W = e.onToggleY,
            N = e.onHover,
            X = e.onHoverEnd,
            B = e.onMove,
            M = e.ignoreCheck,
            $ = e.isNormalizer,
            J = e.onGestureStart,
            m = e.onGestureEnd,
            tt = e.onWheel,
            qt = e.onEnable,
            Le = e.onDisable,
            ft = e.onClick,
            Rt = e.scrollSpeed,
            $t = e.capture,
            yt = e.allowClicks,
            jt = e.lockAxis,
            zt = e.onLockAxis;
        this.target = a = se(a) || Si, this.vars = e, f && (f = Bt.utils.toArray(f)), i = i || 1e-9, n = n || 0, _ = _ || 1, Rt = Rt || 1, s = s || "wheel,touch,pointer", c = c !== !1, l || (l = parseFloat(we.getComputedStyle(br).lineHeight) || 22);
        var di, Gt, Ie, Z, pt, ne, fe, x = this,
            he = 0,
            ei = 0,
            pi = e.passive || !u,
            wt = Ai(a, ie),
            _i = Ai(a, Et),
            Di = wt(),
            nr = _i(),
            Ot = ~s.indexOf("touch") && !~s.indexOf("pointer") && We[0] === "pointerdown",
            gi = Jr(a),
            _t = a.ownerDocument || Ti,
            Be = [0, 0, 0],
            ke = [0, 0, 0],
            ii = 0,
            Mr = function() {
                return ii = Qr()
            },
            bt = function(I, Q) {
                return (x.event = I) && f && ~f.indexOf(I.target) || Q && Ot && I.pointerType !== "touch" || M && M(I, Q)
            },
            mn = function() {
                x._vx.reset(), x._vy.reset(), Gt.pause(), h && h(x)
            },
            mi = function() {
                var I = x.deltaX = qo(Be),
                    Q = x.deltaY = qo(ke),
                    A = Math.abs(I) >= i,
                    Y = Math.abs(Q) >= i;
                L && (A || Y) && L(x, I, Q, Be, ke), A && (y && x.deltaX > 0 && y(x), S && x.deltaX < 0 && S(x), k && k(x), P && x.deltaX < 0 != he < 0 && P(x), he = x.deltaX, Be[0] = Be[1] = Be[2] = 0), Y && (C && x.deltaY > 0 && C(x), w && x.deltaY < 0 && w(x), O && O(x), W && x.deltaY < 0 != ei < 0 && W(x), ei = x.deltaY, ke[0] = ke[1] = ke[2] = 0), (Z || Ie) && (B && B(x), Ie && (b(x), Ie = !1), Z = !1), ne && !(ne = !1) && zt && zt(x), pt && (tt(x), pt = !1), di = 0
            },
            sr = function(I, Q, A) {
                Be[A] += I, ke[A] += Q, x._vx.update(I), x._vy.update(Q), c ? di || (di = requestAnimationFrame(mi)) : mi()
            },
            or = function(I, Q) {
                jt && !fe && (x.axis = fe = Math.abs(I) > Math.abs(Q) ? "x" : "y", ne = !0), fe !== "y" && (Be[2] += I, x._vx.update(I, !0)), fe !== "x" && (ke[2] += Q, x._vy.update(Q, !0)), c ? di || (di = requestAnimationFrame(mi)) : mi()
            },
            vi = function(I) {
                if (!bt(I, 1)) {
                    I = Fr(I, u);
                    var Q = I.clientX,
                        A = I.clientY,
                        Y = Q - x.x,
                        R = A - x.y,
                        z = x.isDragging;
                    x.x = Q, x.y = A, (z || Math.abs(x.startX - Q) >= n || Math.abs(x.startY - A) >= n) && (b && (Ie = !0), z || (x.isDragging = !0), or(Y, R), z || v && v(x))
                }
            },
            Ri = x.onPress = function(F) {
                bt(F, 1) || F && F.button || (x.axis = fe = null, Gt.pause(), x.isPressed = !0, F = Fr(F), he = ei = 0, x.startX = x.x = F.clientX, x.startY = x.y = F.clientY, x._vx.reset(), x._vy.reset(), Zt($ ? a : _t, We[1], vi, pi, !0), x.deltaX = x.deltaY = 0, E && E(x))
            },
            U = x.onRelease = function(F) {
                if (!bt(F, 1)) {
                    Kt($ ? a : _t, We[1], vi, !0);
                    var I = !isNaN(x.y - x.startY),
                        Q = x.isDragging,
                        A = Q && (Math.abs(x.x - x.startX) > 3 || Math.abs(x.y - x.startY) > 3),
                        Y = Fr(F);
                    !A && I && (x._vx.reset(), x._vy.reset(), u && yt && Bt.delayedCall(.08, function() {
                        if (Qr() - ii > 300 && !F.defaultPrevented) {
                            if (F.target.click) F.target.click();
                            else if (_t.createEvent) {
                                var R = _t.createEvent("MouseEvents");
                                R.initMouseEvent("click", !0, !0, we, 1, Y.screenX, Y.screenY, Y.clientX, Y.clientY, !1, !1, !1, !1, 0, null), F.target.dispatchEvent(R)
                            }
                        }
                    })), x.isDragging = x.isGesturing = x.isPressed = !1, h && Q && !$ && Gt.restart(!0), T && Q && T(x), g && g(x, A)
                }
            },
            Li = function(I) {
                return I.touches && I.touches.length > 1 && (x.isGesturing = !0) && J(I, x.isDragging)
            },
            ze = function() {
                return (x.isGesturing = !1) || m(x)
            },
            Fe = function(I) {
                if (!bt(I)) {
                    var Q = wt(),
                        A = _i();
                    sr((Q - Di) * Rt, (A - nr) * Rt, 1), Di = Q, nr = A, h && Gt.restart(!0)
                }
            },
            Ye = function(I) {
                if (!bt(I)) {
                    I = Fr(I, u), tt && (pt = !0);
                    var Q = (I.deltaMode === 1 ? l : I.deltaMode === 2 ? we.innerHeight : 1) * _;
                    sr(I.deltaX * Q, I.deltaY * Q, 0), h && !$ && Gt.restart(!0)
                }
            },
            Ii = function(I) {
                if (!bt(I)) {
                    var Q = I.clientX,
                        A = I.clientY,
                        Y = Q - x.x,
                        R = A - x.y;
                    x.x = Q, x.y = A, Z = !0, h && Gt.restart(!0), (Y || R) && or(Y, R)
                }
            },
            ar = function(I) {
                x.event = I, N(x)
            },
            ri = function(I) {
                x.event = I, X(x)
            },
            Dr = function(I) {
                return bt(I) || Fr(I, u) && ft(x)
            };
        Gt = x._dc = Bt.delayedCall(d || .25, mn).pause(), x.deltaX = x.deltaY = 0, x._vx = $s(0, 50, !0), x._vy = $s(0, 50, !0), x.scrollX = wt, x.scrollY = _i, x.isDragging = x.isGesturing = x.isPressed = !1, Sl(this), x.enable = function(F) {
            return x.isEnabled || (Zt(gi ? _t : a, "scroll", qs), s.indexOf("scroll") >= 0 && Zt(gi ? _t : a, "scroll", Fe, pi, $t), s.indexOf("wheel") >= 0 && Zt(a, "wheel", Ye, pi, $t), (s.indexOf("touch") >= 0 && xl || s.indexOf("pointer") >= 0) && (Zt(a, We[0], Ri, pi, $t), Zt(_t, We[2], U), Zt(_t, We[3], U), yt && Zt(a, "click", Mr, !0, !0), ft && Zt(a, "click", Dr), J && Zt(_t, "gesturestart", Li), m && Zt(_t, "gestureend", ze), N && Zt(a, Ni + "enter", ar), X && Zt(a, Ni + "leave", ri), B && Zt(a, Ni + "move", Ii)), x.isEnabled = !0, F && F.type && Ri(F), qt && qt(x)), x
        }, x.disable = function() {
            x.isEnabled && (gr.filter(function(F) {
                return F !== x && Jr(F.target)
            }).length || Kt(gi ? _t : a, "scroll", qs), x.isPressed && (x._vx.reset(), x._vy.reset(), Kt($ ? a : _t, We[1], vi, !0)), Kt(gi ? _t : a, "scroll", Fe, $t), Kt(a, "wheel", Ye, $t), Kt(a, We[0], Ri, $t), Kt(_t, We[2], U), Kt(_t, We[3], U), Kt(a, "click", Mr, !0), Kt(a, "click", Dr), Kt(_t, "gesturestart", Li), Kt(_t, "gestureend", ze), Kt(a, Ni + "enter", ar), Kt(a, Ni + "leave", ri), Kt(a, Ni + "move", Ii), x.isEnabled = x.isPressed = x.isDragging = !1, Le && Le(x))
        }, x.kill = x.revert = function() {
            x.disable();
            var F = gr.indexOf(x);
            F >= 0 && gr.splice(F, 1), li === x && (li = 0)
        }, gr.push(x), $ && Jr(a) && (li = x), x.enable(p)
    }, $u(o, [{
        key: "velocityX",
        get: function() {
            return this._vx.getVelocity()
        }
    }, {
        key: "velocityY",
        get: function() {
            return this._vy.getVelocity()
        }
    }]), o
}();
vt.version = "3.12.5";
vt.create = function(o) {
    return new vt(o)
};
vt.register = Cl;
vt.getAll = function() {
    return gr.slice()
};
vt.getById = function(o) {
    return gr.filter(function(t) {
        return t.vars.id === o
    })[0]
};
kl() && Bt.registerPlugin(vt);
/*!
 * ScrollTrigger 3.12.5
 * https://gsap.com
 *
 * @license Copyright 2008-2024, GreenSock. All rights reserved.
 * Subject to the terms at https://gsap.com/standard-license or for
 * Club GSAP members, the agreement issued with that membership.
 * @author: Jack Doyle, jack@greensock.com
 */
var D, fr, G, lt, He, nt, Pl, Qn, pn, tn, Vr, Sn, Wt, as, js, Jt, $o, jo, hr, Al, bs, Ml, Qt, Gs, Dl, Rl, yi, Ks, wo, xr, bo, Jn, Zs, xs, kn = 1,
    Ht = Date.now,
    Ts = Ht(),
    De = 0,
    Xr = 0,
    Go = function(t, r, e) {
        var i = ge(t) && (t.substr(0, 6) === "clamp(" || t.indexOf("max") > -1);
        return e["_" + r + "Clamp"] = i, i ? t.substr(6, t.length - 7) : t
    },
    Ko = function(t, r) {
        return r && (!ge(t) || t.substr(0, 6) !== "clamp(") ? "clamp(" + t + ")" : t
    },
    Gu = function o() {
        return Xr && requestAnimationFrame(o)
    },
    Zo = function() {
        return as = 1
    },
    Qo = function() {
        return as = 0
    },
    je = function(t) {
        return t
    },
    Ur = function(t) {
        return Math.round(t * 1e5) / 1e5 || 0
    },
    Ll = function() {
        return typeof window < "u"
    },
    Il = function() {
        return D || Ll() && (D = window.gsap) && D.registerPlugin && D
    },
    tr = function(t) {
        return !!~Pl.indexOf(t)
    },
    Bl = function(t) {
        return (t === "Height" ? bo : G["inner" + t]) || He["client" + t] || nt["client" + t]
    },
    zl = function(t) {
        return Oi(t, "getBoundingClientRect") || (tr(t) ? function() {
            return Hn.width = G.innerWidth, Hn.height = bo, Hn
        } : function() {
            return ai(t)
        })
    },
    Ku = function(t, r, e) {
        var i = e.d,
            n = e.d2,
            s = e.a;
        return (s = Oi(t, "getBoundingClientRect")) ? function() {
            return s()[i]
        } : function() {
            return (r ? Bl(n) : t["client" + n]) || 0
        }
    },
    Zu = function(t, r) {
        return !r || ~Je.indexOf(t) ? zl(t) : function() {
            return Hn
        }
    },
    Ze = function(t, r) {
        var e = r.s,
            i = r.d2,
            n = r.d,
            s = r.a;
        return Math.max(0, (e = "scroll" + i) && (s = Oi(t, e)) ? s() - zl(t)()[n] : tr(t) ? (He[e] || nt[e]) - Bl(i) : t[e] - t["offset" + i])
    },
    En = function(t, r) {
        for (var e = 0; e < hr.length; e += 3)(!r || ~r.indexOf(hr[e + 1])) && t(hr[e], hr[e + 1], hr[e + 2])
    },
    ge = function(t) {
        return typeof t == "string"
    },
    re = function(t) {
        return typeof t == "function"
    },
    qr = function(t) {
        return typeof t == "number"
    },
    Wi = function(t) {
        return typeof t == "object"
    },
    Yr = function(t, r, e) {
        return t && t.progress(r ? 0 : 1) && e && t.pause()
    },
    Ss = function(t, r) {
        if (t.enabled) {
            var e = t._ctx ? t._ctx.add(function() {
                return r(t)
            }) : r(t);
            e && e.totalTime && (t.callbackAnimation = e)
        }
    },
    cr = Math.abs,
    Fl = "left",
    Yl = "top",
    xo = "right",
    To = "bottom",
    Gi = "width",
    Ki = "height",
    en = "Right",
    rn = "Left",
    nn = "Top",
    sn = "Bottom",
    xt = "padding",
    Ce = "margin",
    Ar = "Width",
    So = "Height",
    kt = "px",
    Pe = function(t) {
        return G.getComputedStyle(t)
    },
    Qu = function(t) {
        var r = Pe(t).position;
        t.style.position = r === "absolute" || r === "fixed" ? r : "relative"
    },
    Jo = function(t, r) {
        for (var e in r) e in t || (t[e] = r[e]);
        return t
    },
    ai = function(t, r) {
        var e = r && Pe(t)[js] !== "matrix(1, 0, 0, 1, 0, 0)" && D.to(t, {
                x: 0,
                y: 0,
                xPercent: 0,
                yPercent: 0,
                rotation: 0,
                rotationX: 0,
                rotationY: 0,
                scale: 1,
                skewX: 0,
                skewY: 0
            }).progress(1),
            i = t.getBoundingClientRect();
        return e && e.progress(0).kill(), i
    },
    ts = function(t, r) {
        var e = r.d2;
        return t["offset" + e] || t["client" + e] || 0
    },
    Nl = function(t) {
        var r = [],
            e = t.labels,
            i = t.duration(),
            n;
        for (n in e) r.push(e[n] / i);
        return r
    },
    Ju = function(t) {
        return function(r) {
            return D.utils.snap(Nl(t), r)
        }
    },
    ko = function(t) {
        var r = D.utils.snap(t),
            e = Array.isArray(t) && t.slice(0).sort(function(i, n) {
                return i - n
            });
        return e ? function(i, n, s) {
            s === void 0 && (s = .001);
            var a;
            if (!n) return r(i);
            if (n > 0) {
                for (i -= s, a = 0; a < e.length; a++)
                    if (e[a] >= i) return e[a];
                return e[a - 1]
            } else
                for (a = e.length, i += s; a--;)
                    if (e[a] <= i) return e[a];
            return e[0]
        } : function(i, n, s) {
            s === void 0 && (s = .001);
            var a = r(i);
            return !n || Math.abs(a - i) < s || a - i < 0 == n < 0 ? a : r(n < 0 ? i - t : i + t)
        }
    },
    tf = function(t) {
        return function(r, e) {
            return ko(Nl(t))(r, e.direction)
        }
    },
    On = function(t, r, e, i) {
        return e.split(",").forEach(function(n) {
            return t(r, n, i)
        })
    },
    At = function(t, r, e, i, n) {
        return t.addEventListener(r, e, {
            passive: !i,
            capture: !!n
        })
    },
    Pt = function(t, r, e, i) {
        return t.removeEventListener(r, e, !!i)
    },
    Cn = function(t, r, e) {
        e = e && e.wheelHandler, e && (t(r, "wheel", e), t(r, "touchmove", e))
    },
    ta = {
        startColor: "green",
        endColor: "red",
        indent: 0,
        fontSize: "16px",
        fontWeight: "normal"
    },
    Pn = {
        toggleActions: "play",
        anticipatePin: 0
    },
    es = {
        top: 0,
        left: 0,
        center: .5,
        bottom: 1,
        right: 1
    },
    Fn = function(t, r) {
        if (ge(t)) {
            var e = t.indexOf("="),
                i = ~e ? +(t.charAt(e - 1) + 1) * parseFloat(t.substr(e + 1)) : 0;
            ~e && (t.indexOf("%") > e && (i *= r / 100), t = t.substr(0, e - 1)), t = i + (t in es ? es[t] * r : ~t.indexOf("%") ? parseFloat(t) * r / 100 : parseFloat(t) || 0)
        }
        return t
    },
    An = function(t, r, e, i, n, s, a, l) {
        var c = n.startColor,
            u = n.endColor,
            h = n.fontSize,
            d = n.indent,
            f = n.fontWeight,
            _ = lt.createElement("div"),
            p = tr(e) || Oi(e, "pinType") === "fixed",
            v = t.indexOf("scroller") !== -1,
            T = p ? nt : e,
            b = t.indexOf("start") !== -1,
            E = b ? c : u,
            g = "border-color:" + E + ";font-size:" + h + ";color:" + E + ";font-weight:" + f + ";pointer-events:none;white-space:nowrap;font-family:sans-serif,Arial;z-index:1000;padding:4px 8px;border-width:0;border-style:solid;";
        return g += "position:" + ((v || l) && p ? "fixed;" : "absolute;"), (v || l || !p) && (g += (i === Et ? xo : To) + ":" + (s + parseFloat(d)) + "px;"), a && (g += "box-sizing:border-box;text-align:left;width:" + a.offsetWidth + "px;"), _._isStart = b, _.setAttribute("class", "gsap-marker-" + t + (r ? " marker-" + r : "")), _.style.cssText = g, _.innerText = r || r === 0 ? t + "-" + r : t, T.children[0] ? T.insertBefore(_, T.children[0]) : T.appendChild(_), _._offset = _["offset" + i.op.d2], Yn(_, 0, i, b), _
    },
    Yn = function(t, r, e, i) {
        var n = {
                display: "block"
            },
            s = e[i ? "os2" : "p2"],
            a = e[i ? "p2" : "os2"];
        t._isFlipped = i, n[e.a + "Percent"] = i ? -100 : 0, n[e.a] = i ? "1px" : 0, n["border" + s + Ar] = 1, n["border" + a + Ar] = 0, n[e.p] = r + "px", D.set(t, n)
    },
    H = [],
    Qs = {},
    _n, ea = function() {
        return Ht() - De > 34 && (_n || (_n = requestAnimationFrame(ui)))
    },
    ur = function() {
        (!Qt || !Qt.isPressed || Qt.startX > nt.clientWidth) && (q.cache++, Qt ? _n || (_n = requestAnimationFrame(ui)) : ui(), De || ir("scrollStart"), De = Ht())
    },
    ks = function() {
        Rl = G.innerWidth, Dl = G.innerHeight
    },
    $r = function() {
        q.cache++, !Wt && !Ml && !lt.fullscreenElement && !lt.webkitFullscreenElement && (!Gs || Rl !== G.innerWidth || Math.abs(G.innerHeight - Dl) > G.innerHeight * .25) && Qn.restart(!0)
    },
    er = {},
    ef = [],
    Wl = function o() {
        return Pt(V, "scrollEnd", o) || Vi(!0)
    },
    ir = function(t) {
        return er[t] && er[t].map(function(r) {
            return r()
        }) || ef
    },
    _e = [],
    Hl = function(t) {
        for (var r = 0; r < _e.length; r += 5)(!t || _e[r + 4] && _e[r + 4].query === t) && (_e[r].style.cssText = _e[r + 1], _e[r].getBBox && _e[r].setAttribute("transform", _e[r + 2] || ""), _e[r + 3].uncache = 1)
    },
    Eo = function(t, r) {
        var e;
        for (Jt = 0; Jt < H.length; Jt++) e = H[Jt], e && (!r || e._ctx === r) && (t ? e.kill(1) : e.revert(!0, !0));
        Jn = !0, r && Hl(r), r || ir("revert")
    },
    Vl = function(t, r) {
        q.cache++, (r || !te) && q.forEach(function(e) {
            return re(e) && e.cacheID++ && (e.rec = 0)
        }), ge(t) && (G.history.scrollRestoration = wo = t)
    },
    te, Zi = 0,
    ia, rf = function() {
        if (ia !== Zi) {
            var t = ia = Zi;
            requestAnimationFrame(function() {
                return t === Zi && Vi(!0)
            })
        }
    },
    Xl = function() {
        nt.appendChild(xr), bo = !Qt && xr.offsetHeight || G.innerHeight, nt.removeChild(xr)
    },
    ra = function(t) {
        return pn(".gsap-marker-start, .gsap-marker-end, .gsap-marker-scroller-start, .gsap-marker-scroller-end").forEach(function(r) {
            return r.style.display = t ? "none" : "block"
        })
    },
    Vi = function(t, r) {
        if (De && !t && !Jn) {
            At(V, "scrollEnd", Wl);
            return
        }
        Xl(), te = V.isRefreshing = !0, q.forEach(function(i) {
            return re(i) && ++i.cacheID && (i.rec = i())
        });
        var e = ir("refreshInit");
        Al && V.sort(), r || Eo(), q.forEach(function(i) {
            re(i) && (i.smooth && (i.target.style.scrollBehavior = "auto"), i(0))
        }), H.slice(0).forEach(function(i) {
            return i.refresh()
        }), Jn = !1, H.forEach(function(i) {
            if (i._subPinOffset && i.pin) {
                var n = i.vars.horizontal ? "offsetWidth" : "offsetHeight",
                    s = i.pin[n];
                i.revert(!0, 1), i.adjustPinSpacing(i.pin[n] - s), i.refresh()
            }
        }), Zs = 1, ra(!0), H.forEach(function(i) {
            var n = Ze(i.scroller, i._dir),
                s = i.vars.end === "max" || i._endClamp && i.end > n,
                a = i._startClamp && i.start >= n;
            (s || a) && i.setPositions(a ? n - 1 : i.start, s ? Math.max(a ? n : i.start + 1, n) : i.end, !0)
        }), ra(!1), Zs = 0, e.forEach(function(i) {
            return i && i.render && i.render(-1)
        }), q.forEach(function(i) {
            re(i) && (i.smooth && requestAnimationFrame(function() {
                return i.target.style.scrollBehavior = "smooth"
            }), i.rec && i(i.rec))
        }), Vl(wo, 1), Qn.pause(), Zi++, te = 2, ui(2), H.forEach(function(i) {
            return re(i.vars.onRefresh) && i.vars.onRefresh(i)
        }), te = V.isRefreshing = !1, ir("refresh")
    },
    Js = 0,
    Nn = 1,
    on, ui = function(t) {
        if (t === 2 || !te && !Jn) {
            V.isUpdating = !0, on && on.update(0);
            var r = H.length,
                e = Ht(),
                i = e - Ts >= 50,
                n = r && H[0].scroll();
            if (Nn = Js > n ? -1 : 1, te || (Js = n), i && (De && !as && e - De > 200 && (De = 0, ir("scrollEnd")), Vr = Ts, Ts = e), Nn < 0) {
                for (Jt = r; Jt-- > 0;) H[Jt] && H[Jt].update(0, i);
                Nn = 1
            } else
                for (Jt = 0; Jt < r; Jt++) H[Jt] && H[Jt].update(0, i);
            V.isUpdating = !1
        }
        _n = 0
    },
    to = [Fl, Yl, To, xo, Ce + sn, Ce + en, Ce + nn, Ce + rn, "display", "flexShrink", "float", "zIndex", "gridColumnStart", "gridColumnEnd", "gridRowStart", "gridRowEnd", "gridArea", "justifySelf", "alignSelf", "placeSelf", "order"],
    Wn = to.concat([Gi, Ki, "boxSizing", "max" + Ar, "max" + So, "position", Ce, xt, xt + nn, xt + en, xt + sn, xt + rn]),
    nf = function(t, r, e) {
        Tr(e);
        var i = t._gsap;
        if (i.spacerIsNative) Tr(i.spacerState);
        else if (t._gsap.swappedIn) {
            var n = r.parentNode;
            n && (n.insertBefore(t, r), n.removeChild(r))
        }
        t._gsap.swappedIn = !1
    },
    Es = function(t, r, e, i) {
        if (!t._gsap.swappedIn) {
            for (var n = to.length, s = r.style, a = t.style, l; n--;) l = to[n], s[l] = e[l];
            s.position = e.position === "absolute" ? "absolute" : "relative", e.display === "inline" && (s.display = "inline-block"), a[To] = a[xo] = "auto", s.flexBasis = e.flexBasis || "auto", s.overflow = "visible", s.boxSizing = "border-box", s[Gi] = ts(t, ie) + kt, s[Ki] = ts(t, Et) + kt, s[xt] = a[Ce] = a[Yl] = a[Fl] = "0", Tr(i), a[Gi] = a["max" + Ar] = e[Gi], a[Ki] = a["max" + So] = e[Ki], a[xt] = e[xt], t.parentNode !== r && (t.parentNode.insertBefore(r, t), r.appendChild(t)), t._gsap.swappedIn = !0
        }
    },
    sf = /([A-Z])/g,
    Tr = function(t) {
        if (t) {
            var r = t.t.style,
                e = t.length,
                i = 0,
                n, s;
            for ((t.t._gsap || D.core.getCache(t.t)).uncache = 1; i < e; i += 2) s = t[i + 1], n = t[i], s ? r[n] = s : r[n] && r.removeProperty(n.replace(sf, "-$1").toLowerCase())
        }
    },
    Mn = function(t) {
        for (var r = Wn.length, e = t.style, i = [], n = 0; n < r; n++) i.push(Wn[n], e[Wn[n]]);
        return i.t = t, i
    },
    of = function(t, r, e) {
        for (var i = [], n = t.length, s = e ? 8 : 0, a; s < n; s += 2) a = t[s], i.push(a, a in r ? r[a] : t[s + 1]);
        return i.t = t.t, i
    },
    Hn = {
        left: 0,
        top: 0
    },
    na = function(t, r, e, i, n, s, a, l, c, u, h, d, f, _) {
        re(t) && (t = t(l)), ge(t) && t.substr(0, 3) === "max" && (t = d + (t.charAt(4) === "=" ? Fn("0" + t.substr(3), e) : 0));
        var p = f ? f.time() : 0,
            v, T, b;
        if (f && f.seek(0), isNaN(t) || (t = +t), qr(t)) f && (t = D.utils.mapRange(f.scrollTrigger.start, f.scrollTrigger.end, 0, d, t)), a && Yn(a, e, i, !0);
        else {
            re(r) && (r = r(l));
            var E = (t || "0").split(" "),
                g, y, S, w;
            b = se(r, l) || nt, g = ai(b) || {}, (!g || !g.left && !g.top) && Pe(b).display === "none" && (w = b.style.display, b.style.display = "block", g = ai(b), w ? b.style.display = w : b.style.removeProperty("display")), y = Fn(E[0], g[i.d]), S = Fn(E[1] || "0", e), t = g[i.p] - c[i.p] - u + y + n - S, a && Yn(a, S, i, e - S < 20 || a._isStart && S > 20), e -= e - S
        }
        if (_ && (l[_] = t || -.001, t < 0 && (t = 0)), s) {
            var C = t + e,
                k = s._isStart;
            v = "scroll" + i.d2, Yn(s, C, i, k && C > 20 || !k && (h ? Math.max(nt[v], He[v]) : s.parentNode[v]) <= C + 1), h && (c = ai(a), h && (s.style[i.op.p] = c[i.op.p] - i.op.m - s._offset + kt))
        }
        return f && b && (v = ai(b), f.seek(d), T = ai(b), f._caScrollDist = v[i.p] - T[i.p], t = t / f._caScrollDist * d), f && f.seek(p), f ? t : Math.round(t)
    },
    af = /(webkit|moz|length|cssText|inset)/i,
    sa = function(t, r, e, i) {
        if (t.parentNode !== r) {
            var n = t.style,
                s, a;
            if (r === nt) {
                t._stOrig = n.cssText, a = Pe(t);
                for (s in a) !+s && !af.test(s) && a[s] && typeof n[s] == "string" && s !== "0" && (n[s] = a[s]);
                n.top = e, n.left = i
            } else n.cssText = t._stOrig;
            D.core.getCache(t).uncache = 1, r.appendChild(t)
        }
    },
    Ul = function(t, r, e) {
        var i = r,
            n = i;
        return function(s) {
            var a = Math.round(t());
            return a !== i && a !== n && Math.abs(a - i) > 3 && Math.abs(a - n) > 3 && (s = a, e && e()), n = i, i = s, s
        }
    },
    Dn = function(t, r, e) {
        var i = {};
        i[r.p] = "+=" + e, D.set(t, i)
    },
    oa = function(t, r) {
        var e = Ai(t, r),
            i = "_scroll" + r.p2,
            n = function s(a, l, c, u, h) {
                var d = s.tween,
                    f = l.onComplete,
                    _ = {};
                c = c || e();
                var p = Ul(e, c, function() {
                    d.kill(), s.tween = 0
                });
                return h = u && h || 0, u = u || a - c, d && d.kill(), l[i] = a, l.inherit = !1, l.modifiers = _, _[i] = function() {
                    return p(c + u * d.ratio + h * d.ratio * d.ratio)
                }, l.onUpdate = function() {
                    q.cache++, s.tween && ui()
                }, l.onComplete = function() {
                    s.tween = 0, f && f.call(d)
                }, d = s.tween = D.to(t, l), d
            };
        return t[i] = e, e.wheelHandler = function() {
            return n.tween && n.tween.kill() && (n.tween = 0)
        }, At(t, "wheel", e.wheelHandler), V.isTouch && At(t, "touchmove", e.wheelHandler), n
    },
    V = function() {
        function o(r, e) {
            fr || o.register(D) || console.warn("Please gsap.registerPlugin(ScrollTrigger)"), Ks(this), this.init(r, e)
        }
        var t = o.prototype;
        return t.init = function(e, i) {
            if (this.progress = this.start = 0, this.vars && this.kill(!0, !0), !Xr) {
                this.update = this.refresh = this.kill = je;
                return
            }
            e = Jo(ge(e) || qr(e) || e.nodeType ? {
                trigger: e
            } : e, Pn);
            var n = e,
                s = n.onUpdate,
                a = n.toggleClass,
                l = n.id,
                c = n.onToggle,
                u = n.onRefresh,
                h = n.scrub,
                d = n.trigger,
                f = n.pin,
                _ = n.pinSpacing,
                p = n.invalidateOnRefresh,
                v = n.anticipatePin,
                T = n.onScrubComplete,
                b = n.onSnapComplete,
                E = n.once,
                g = n.snap,
                y = n.pinReparent,
                S = n.pinSpacer,
                w = n.containerAnimation,
                C = n.fastScrollEnd,
                k = n.preventOverlaps,
                O = e.horizontal || e.containerAnimation && e.horizontal !== !1 ? ie : Et,
                L = !h && h !== 0,
                P = se(e.scroller || G),
                W = D.core.getCache(P),
                N = tr(P),
                X = ("pinType" in e ? e.pinType : Oi(P, "pinType") || N && "fixed") === "fixed",
                B = [e.onEnter, e.onLeave, e.onEnterBack, e.onLeaveBack],
                M = L && e.toggleActions.split(" "),
                $ = "markers" in e ? e.markers : Pn.markers,
                J = N ? 0 : parseFloat(Pe(P)["border" + O.p2 + Ar]) || 0,
                m = this,
                tt = e.onRefreshInit && function() {
                    return e.onRefreshInit(m)
                },
                qt = Ku(P, N, O),
                Le = Zu(P, N),
                ft = 0,
                Rt = 0,
                $t = 0,
                yt = Ai(P, O),
                jt, zt, di, Gt, Ie, Z, pt, ne, fe, x, he, ei, pi, wt, _i, Di, nr, Ot, gi, _t, Be, ke, ii, Mr, bt, mn, mi, sr, or, vi, Ri, U, Li, ze, Fe, Ye, Ii, ar, ri;
            if (m._startClamp = m._endClamp = !1, m._dir = O, v *= 45, m.scroller = P, m.scroll = w ? w.time.bind(w) : yt, Gt = yt(), m.vars = e, i = i || e.animation, "refreshPriority" in e && (Al = 1, e.refreshPriority === -9999 && (on = m)), W.tweenScroll = W.tweenScroll || {
                    top: oa(P, Et),
                    left: oa(P, ie)
                }, m.tweenTo = jt = W.tweenScroll[O.p], m.scrubDuration = function(A) {
                    Li = qr(A) && A, Li ? U ? U.duration(A) : U = D.to(i, {
                        ease: "expo",
                        totalProgress: "+=0",
                        inherit: !1,
                        duration: Li,
                        paused: !0,
                        onComplete: function() {
                            return T && T(m)
                        }
                    }) : (U && U.progress(1).kill(), U = 0)
                }, i && (i.vars.lazy = !1, i._initted && !m.isReverted || i.vars.immediateRender !== !1 && e.immediateRender !== !1 && i.duration() && i.render(0, !0, !0), m.animation = i.pause(), i.scrollTrigger = m, m.scrubDuration(h), vi = 0, l || (l = i.vars.id)), g && ((!Wi(g) || g.push) && (g = {
                    snapTo: g
                }), "scrollBehavior" in nt.style && D.set(N ? [nt, He] : P, {
                    scrollBehavior: "auto"
                }), q.forEach(function(A) {
                    return re(A) && A.target === (N ? lt.scrollingElement || He : P) && (A.smooth = !1)
                }), di = re(g.snapTo) ? g.snapTo : g.snapTo === "labels" ? Ju(i) : g.snapTo === "labelsDirectional" ? tf(i) : g.directional !== !1 ? function(A, Y) {
                    return ko(g.snapTo)(A, Ht() - Rt < 500 ? 0 : Y.direction)
                } : D.utils.snap(g.snapTo), ze = g.duration || {
                    min: .1,
                    max: 2
                }, ze = Wi(ze) ? tn(ze.min, ze.max) : tn(ze, ze), Fe = D.delayedCall(g.delay || Li / 2 || .1, function() {
                    var A = yt(),
                        Y = Ht() - Rt < 500,
                        R = jt.tween;
                    if ((Y || Math.abs(m.getVelocity()) < 10) && !R && !as && ft !== A) {
                        var z = (A - Z) / wt,
                            Ct = i && !L ? i.totalProgress() : z,
                            j = Y ? 0 : (Ct - Ri) / (Ht() - Vr) * 1e3 || 0,
                            gt = D.utils.clamp(-z, 1 - z, cr(j / 2) * j / .185),
                            Ft = z + (g.inertia === !1 ? 0 : gt),
                            ht, st, et = g,
                            Ne = et.onStart,
                            at = et.onInterrupt,
                            de = et.onComplete;
                        if (ht = di(Ft, m), qr(ht) || (ht = Ft), st = Math.round(Z + ht * wt), A <= pt && A >= Z && st !== A) {
                            if (R && !R._initted && R.data <= cr(st - A)) return;
                            g.inertia === !1 && (gt = ht - z), jt(st, {
                                duration: ze(cr(Math.max(cr(Ft - Ct), cr(ht - Ct)) * .185 / j / .05 || 0)),
                                ease: g.ease || "power3",
                                data: cr(st - A),
                                onInterrupt: function() {
                                    return Fe.restart(!0) && at && at(m)
                                },
                                onComplete: function() {
                                    m.update(), ft = yt(), i && (U ? U.resetTo("totalProgress", ht, i._tTime / i._tDur) : i.progress(ht)), vi = Ri = i && !L ? i.totalProgress() : m.progress, b && b(m), de && de(m)
                                }
                            }, A, gt * wt, st - A - gt * wt), Ne && Ne(m, jt.tween)
                        }
                    } else m.isActive && ft !== A && Fe.restart(!0)
                }).pause()), l && (Qs[l] = m), d = m.trigger = se(d || f !== !0 && f), ri = d && d._gsap && d._gsap.stRevert, ri && (ri = ri(m)), f = f === !0 ? d : se(f), ge(a) && (a = {
                    targets: d,
                    className: a
                }), f && (_ === !1 || _ === Ce || (_ = !_ && f.parentNode && f.parentNode.style && Pe(f.parentNode).display === "flex" ? !1 : xt), m.pin = f, zt = D.core.getCache(f), zt.spacer ? _i = zt.pinState : (S && (S = se(S), S && !S.nodeType && (S = S.current || S.nativeElement), zt.spacerIsNative = !!S, S && (zt.spacerState = Mn(S))), zt.spacer = Ot = S || lt.createElement("div"), Ot.classList.add("pin-spacer"), l && Ot.classList.add("pin-spacer-" + l), zt.pinState = _i = Mn(f)), e.force3D !== !1 && D.set(f, {
                    force3D: !0
                }), m.spacer = Ot = zt.spacer, or = Pe(f), Mr = or[_ + O.os2], _t = D.getProperty(f), Be = D.quickSetter(f, O.a, kt), Es(f, Ot, or), nr = Mn(f)), $) {
                ei = Wi($) ? Jo($, ta) : ta, x = An("scroller-start", l, P, O, ei, 0), he = An("scroller-end", l, P, O, ei, 0, x), gi = x["offset" + O.op.d2];
                var Dr = se(Oi(P, "content") || P);
                ne = this.markerStart = An("start", l, Dr, O, ei, gi, 0, w), fe = this.markerEnd = An("end", l, Dr, O, ei, gi, 0, w), w && (ar = D.quickSetter([ne, fe], O.a, kt)), !X && !(Je.length && Oi(P, "fixedMarkers") === !0) && (Qu(N ? nt : P), D.set([x, he], {
                    force3D: !0
                }), mn = D.quickSetter(x, O.a, kt), sr = D.quickSetter(he, O.a, kt))
            }
            if (w) {
                var F = w.vars.onUpdate,
                    I = w.vars.onUpdateParams;
                w.eventCallback("onUpdate", function() {
                    m.update(0, 0, 1), F && F.apply(w, I || [])
                })
            }
            if (m.previous = function() {
                    return H[H.indexOf(m) - 1]
                }, m.next = function() {
                    return H[H.indexOf(m) + 1]
                }, m.revert = function(A, Y) {
                    if (!Y) return m.kill(!0);
                    var R = A !== !1 || !m.enabled,
                        z = Wt;
                    R !== m.isReverted && (R && (Ye = Math.max(yt(), m.scroll.rec || 0), $t = m.progress, Ii = i && i.progress()), ne && [ne, fe, x, he].forEach(function(Ct) {
                        return Ct.style.display = R ? "none" : "block"
                    }), R && (Wt = m, m.update(R)), f && (!y || !m.isActive) && (R ? nf(f, Ot, _i) : Es(f, Ot, Pe(f), bt)), R || m.update(R), Wt = z, m.isReverted = R)
                }, m.refresh = function(A, Y, R, z) {
                    if (!((Wt || !m.enabled) && !Y)) {
                        if (f && A && De) {
                            At(o, "scrollEnd", Wl);
                            return
                        }!te && tt && tt(m), Wt = m, jt.tween && !R && (jt.tween.kill(), jt.tween = 0), U && U.pause(), p && i && i.revert({
                            kill: !1
                        }).invalidate(), m.isReverted || m.revert(!0, !0), m._subPinOffset = !1;
                        var Ct = qt(),
                            j = Le(),
                            gt = w ? w.duration() : Ze(P, O),
                            Ft = wt <= .01,
                            ht = 0,
                            st = z || 0,
                            et = Wi(R) ? R.end : e.end,
                            Ne = e.endTrigger || d,
                            at = Wi(R) ? R.start : e.start || (e.start === 0 || !d ? 0 : f ? "0 0" : "0 100%"),
                            de = m.pinnedContainer = e.pinnedContainer && se(e.pinnedContainer, m),
                            Xe = d && Math.max(0, H.indexOf(m)) || 0,
                            Lt = Xe,
                            It, Yt, Bi, vn, Nt, St, Ue, ls, Oo, Rr, qe, Lr, yn;
                        for ($ && Wi(R) && (Lr = D.getProperty(x, O.p), yn = D.getProperty(he, O.p)); Lt--;) St = H[Lt], St.end || St.refresh(0, 1) || (Wt = m), Ue = St.pin, Ue && (Ue === d || Ue === f || Ue === de) && !St.isReverted && (Rr || (Rr = []), Rr.unshift(St), St.revert(!0, !0)), St !== H[Lt] && (Xe--, Lt--);
                        for (re(at) && (at = at(m)), at = Go(at, "start", m), Z = na(at, d, Ct, O, yt(), ne, x, m, j, J, X, gt, w, m._startClamp && "_startClamp") || (f ? -.001 : 0), re(et) && (et = et(m)), ge(et) && !et.indexOf("+=") && (~et.indexOf(" ") ? et = (ge(at) ? at.split(" ")[0] : "") + et : (ht = Fn(et.substr(2), Ct), et = ge(at) ? at : (w ? D.utils.mapRange(0, w.duration(), w.scrollTrigger.start, w.scrollTrigger.end, Z) : Z) + ht, Ne = d)), et = Go(et, "end", m), pt = Math.max(Z, na(et || (Ne ? "100% 0" : gt), Ne, Ct, O, yt() + ht, fe, he, m, j, J, X, gt, w, m._endClamp && "_endClamp")) || -.001, ht = 0, Lt = Xe; Lt--;) St = H[Lt], Ue = St.pin, Ue && St.start - St._pinPush <= Z && !w && St.end > 0 && (It = St.end - (m._startClamp ? Math.max(0, St.start) : St.start), (Ue === d && St.start - St._pinPush < Z || Ue === de) && isNaN(at) && (ht += It * (1 - St.progress)), Ue === f && (st += It));
                        if (Z += ht, pt += ht, m._startClamp && (m._startClamp += ht), m._endClamp && !te && (m._endClamp = pt || -.001, pt = Math.min(pt, Ze(P, O))), wt = pt - Z || (Z -= .01) && .001, Ft && ($t = D.utils.clamp(0, 1, D.utils.normalize(Z, pt, Ye))), m._pinPush = st, ne && ht && (It = {}, It[O.a] = "+=" + ht, de && (It[O.p] = "-=" + yt()), D.set([ne, fe], It)), f && !(Zs && m.end >= Ze(P, O))) It = Pe(f), vn = O === Et, Bi = yt(), ke = parseFloat(_t(O.a)) + st, !gt && pt > 1 && (qe = (N ? lt.scrollingElement || He : P).style, qe = {
                            style: qe,
                            value: qe["overflow" + O.a.toUpperCase()]
                        }, N && Pe(nt)["overflow" + O.a.toUpperCase()] !== "scroll" && (qe.style["overflow" + O.a.toUpperCase()] = "scroll")), Es(f, Ot, It), nr = Mn(f), Yt = ai(f, !0), ls = X && Ai(P, vn ? ie : Et)(), _ ? (bt = [_ + O.os2, wt + st + kt], bt.t = Ot, Lt = _ === xt ? ts(f, O) + wt + st : 0, Lt && (bt.push(O.d, Lt + kt), Ot.style.flexBasis !== "auto" && (Ot.style.flexBasis = Lt + kt)), Tr(bt), de && H.forEach(function(Ir) {
                            Ir.pin === de && Ir.vars.pinSpacing !== !1 && (Ir._subPinOffset = !0)
                        }), X && yt(Ye)) : (Lt = ts(f, O), Lt && Ot.style.flexBasis !== "auto" && (Ot.style.flexBasis = Lt + kt)), X && (Nt = {
                            top: Yt.top + (vn ? Bi - Z : ls) + kt,
                            left: Yt.left + (vn ? ls : Bi - Z) + kt,
                            boxSizing: "border-box",
                            position: "fixed"
                        }, Nt[Gi] = Nt["max" + Ar] = Math.ceil(Yt.width) + kt, Nt[Ki] = Nt["max" + So] = Math.ceil(Yt.height) + kt, Nt[Ce] = Nt[Ce + nn] = Nt[Ce + en] = Nt[Ce + sn] = Nt[Ce + rn] = "0", Nt[xt] = It[xt], Nt[xt + nn] = It[xt + nn], Nt[xt + en] = It[xt + en], Nt[xt + sn] = It[xt + sn], Nt[xt + rn] = It[xt + rn], Di = of(_i, Nt, y), te && yt(0)), i ? (Oo = i._initted, bs(1), i.render(i.duration(), !0, !0), ii = _t(O.a) - ke + wt + st, mi = Math.abs(wt - ii) > 1, X && mi && Di.splice(Di.length - 2, 2), i.render(0, !0, !0), Oo || i.invalidate(!0), i.parent || i.totalTime(i.totalTime()), bs(0)) : ii = wt, qe && (qe.value ? qe.style["overflow" + O.a.toUpperCase()] = qe.value : qe.style.removeProperty("overflow-" + O.a));
                        else if (d && yt() && !w)
                            for (Yt = d.parentNode; Yt && Yt !== nt;) Yt._pinOffset && (Z -= Yt._pinOffset, pt -= Yt._pinOffset), Yt = Yt.parentNode;
                        Rr && Rr.forEach(function(Ir) {
                            return Ir.revert(!1, !0)
                        }), m.start = Z, m.end = pt, Gt = Ie = te ? Ye : yt(), !w && !te && (Gt < Ye && yt(Ye), m.scroll.rec = 0), m.revert(!1, !0), Rt = Ht(), Fe && (ft = -1, Fe.restart(!0)), Wt = 0, i && L && (i._initted || Ii) && i.progress() !== Ii && i.progress(Ii || 0, !0).render(i.time(), !0, !0), (Ft || $t !== m.progress || w || p) && (i && !L && i.totalProgress(w && Z < -.001 && !$t ? D.utils.normalize(Z, pt, 0) : $t, !0), m.progress = Ft || (Gt - Z) / wt === $t ? 0 : $t), f && _ && (Ot._pinOffset = Math.round(m.progress * ii)), U && U.invalidate(), isNaN(Lr) || (Lr -= D.getProperty(x, O.p), yn -= D.getProperty(he, O.p), Dn(x, O, Lr), Dn(ne, O, Lr - (z || 0)), Dn(he, O, yn), Dn(fe, O, yn - (z || 0))), Ft && !te && m.update(), u && !te && !pi && (pi = !0, u(m), pi = !1)
                    }
                }, m.getVelocity = function() {
                    return (yt() - Ie) / (Ht() - Vr) * 1e3 || 0
                }, m.endAnimation = function() {
                    Yr(m.callbackAnimation), i && (U ? U.progress(1) : i.paused() ? L || Yr(i, m.direction < 0, 1) : Yr(i, i.reversed()))
                }, m.labelToScroll = function(A) {
                    return i && i.labels && (Z || m.refresh() || Z) + i.labels[A] / i.duration() * wt || 0
                }, m.getTrailing = function(A) {
                    var Y = H.indexOf(m),
                        R = m.direction > 0 ? H.slice(0, Y).reverse() : H.slice(Y + 1);
                    return (ge(A) ? R.filter(function(z) {
                        return z.vars.preventOverlaps === A
                    }) : R).filter(function(z) {
                        return m.direction > 0 ? z.end <= Z : z.start >= pt
                    })
                }, m.update = function(A, Y, R) {
                    if (!(w && !R && !A)) {
                        var z = te === !0 ? Ye : m.scroll(),
                            Ct = A ? 0 : (z - Z) / wt,
                            j = Ct < 0 ? 0 : Ct > 1 ? 1 : Ct || 0,
                            gt = m.progress,
                            Ft, ht, st, et, Ne, at, de, Xe;
                        if (Y && (Ie = Gt, Gt = w ? yt() : z, g && (Ri = vi, vi = i && !L ? i.totalProgress() : j)), v && f && !Wt && !kn && De && (!j && Z < z + (z - Ie) / (Ht() - Vr) * v ? j = 1e-4 : j === 1 && pt > z + (z - Ie) / (Ht() - Vr) * v && (j = .9999)), j !== gt && m.enabled) {
                            if (Ft = m.isActive = !!j && j < 1, ht = !!gt && gt < 1, at = Ft !== ht, Ne = at || !!j != !!gt, m.direction = j > gt ? 1 : -1, m.progress = j, Ne && !Wt && (st = j && !gt ? 0 : j === 1 ? 1 : gt === 1 ? 2 : 3, L && (et = !at && M[st + 1] !== "none" && M[st + 1] || M[st], Xe = i && (et === "complete" || et === "reset" || et in i))), k && (at || Xe) && (Xe || h || !i) && (re(k) ? k(m) : m.getTrailing(k).forEach(function(Bi) {
                                    return Bi.endAnimation()
                                })), L || (U && !Wt && !kn ? (U._dp._time - U._start !== U._time && U.render(U._dp._time - U._start), U.resetTo ? U.resetTo("totalProgress", j, i._tTime / i._tDur) : (U.vars.totalProgress = j, U.invalidate().restart())) : i && i.totalProgress(j, !!(Wt && (Rt || A)))), f) {
                                if (A && _ && (Ot.style[_ + O.os2] = Mr), !X) Be(Ur(ke + ii * j));
                                else if (Ne) {
                                    if (de = !A && j > gt && pt + 1 > z && z + 1 >= Ze(P, O), y)
                                        if (!A && (Ft || de)) {
                                            var Lt = ai(f, !0),
                                                It = z - Z;
                                            sa(f, nt, Lt.top + (O === Et ? It : 0) + kt, Lt.left + (O === Et ? 0 : It) + kt)
                                        } else sa(f, Ot);
                                    Tr(Ft || de ? Di : nr), mi && j < 1 && Ft || Be(ke + (j === 1 && !de ? ii : 0))
                                }
                            }
                            g && !jt.tween && !Wt && !kn && Fe.restart(!0), a && (at || E && j && (j < 1 || !xs)) && pn(a.targets).forEach(function(Bi) {
                                return Bi.classList[Ft || E ? "add" : "remove"](a.className)
                            }), s && !L && !A && s(m), Ne && !Wt ? (L && (Xe && (et === "complete" ? i.pause().totalProgress(1) : et === "reset" ? i.restart(!0).pause() : et === "restart" ? i.restart(!0) : i[et]()), s && s(m)), (at || !xs) && (c && at && Ss(m, c), B[st] && Ss(m, B[st]), E && (j === 1 ? m.kill(!1, 1) : B[st] = 0), at || (st = j === 1 ? 1 : 3, B[st] && Ss(m, B[st]))), C && !Ft && Math.abs(m.getVelocity()) > (qr(C) ? C : 2500) && (Yr(m.callbackAnimation), U ? U.progress(1) : Yr(i, et === "reverse" ? 1 : !j, 1))) : L && s && !Wt && s(m)
                        }
                        if (sr) {
                            var Yt = w ? z / w.duration() * (w._caScrollDist || 0) : z;
                            mn(Yt + (x._isFlipped ? 1 : 0)), sr(Yt)
                        }
                        ar && ar(-z / w.duration() * (w._caScrollDist || 0))
                    }
                }, m.enable = function(A, Y) {
                    m.enabled || (m.enabled = !0, At(P, "resize", $r), N || At(P, "scroll", ur), tt && At(o, "refreshInit", tt), A !== !1 && (m.progress = $t = 0, Gt = Ie = ft = yt()), Y !== !1 && m.refresh())
                }, m.getTween = function(A) {
                    return A && jt ? jt.tween : U
                }, m.setPositions = function(A, Y, R, z) {
                    if (w) {
                        var Ct = w.scrollTrigger,
                            j = w.duration(),
                            gt = Ct.end - Ct.start;
                        A = Ct.start + gt * A / j, Y = Ct.start + gt * Y / j
                    }
                    m.refresh(!1, !1, {
                        start: Ko(A, R && !!m._startClamp),
                        end: Ko(Y, R && !!m._endClamp)
                    }, z), m.update()
                }, m.adjustPinSpacing = function(A) {
                    if (bt && A) {
                        var Y = bt.indexOf(O.d) + 1;
                        bt[Y] = parseFloat(bt[Y]) + A + kt, bt[1] = parseFloat(bt[1]) + A + kt, Tr(bt)
                    }
                }, m.disable = function(A, Y) {
                    if (m.enabled && (A !== !1 && m.revert(!0, !0), m.enabled = m.isActive = !1, Y || U && U.pause(), Ye = 0, zt && (zt.uncache = 1), tt && Pt(o, "refreshInit", tt), Fe && (Fe.pause(), jt.tween && jt.tween.kill() && (jt.tween = 0)), !N)) {
                        for (var R = H.length; R--;)
                            if (H[R].scroller === P && H[R] !== m) return;
                        Pt(P, "resize", $r), N || Pt(P, "scroll", ur)
                    }
                }, m.kill = function(A, Y) {
                    m.disable(A, Y), U && !Y && U.kill(), l && delete Qs[l];
                    var R = H.indexOf(m);
                    R >= 0 && H.splice(R, 1), R === Jt && Nn > 0 && Jt--, R = 0, H.forEach(function(z) {
                        return z.scroller === m.scroller && (R = 1)
                    }), R || te || (m.scroll.rec = 0), i && (i.scrollTrigger = null, A && i.revert({
                        kill: !1
                    }), Y || i.kill()), ne && [ne, fe, x, he].forEach(function(z) {
                        return z.parentNode && z.parentNode.removeChild(z)
                    }), on === m && (on = 0), f && (zt && (zt.uncache = 1), R = 0, H.forEach(function(z) {
                        return z.pin === f && R++
                    }), R || (zt.spacer = 0)), e.onKill && e.onKill(m)
                }, H.push(m), m.enable(!1, !1), ri && ri(m), i && i.add && !wt) {
                var Q = m.update;
                m.update = function() {
                    m.update = Q, Z || pt || m.refresh()
                }, D.delayedCall(.01, m.update), wt = .01, Z = pt = 0
            } else m.refresh();
            f && rf()
        }, o.register = function(e) {
            return fr || (D = e || Il(), Ll() && window.document && o.enable(), fr = Xr), fr
        }, o.defaults = function(e) {
            if (e)
                for (var i in e) Pn[i] = e[i];
            return Pn
        }, o.disable = function(e, i) {
            Xr = 0, H.forEach(function(s) {
                return s[i ? "kill" : "disable"](e)
            }), Pt(G, "wheel", ur), Pt(lt, "scroll", ur), clearInterval(Sn), Pt(lt, "touchcancel", je), Pt(nt, "touchstart", je), On(Pt, lt, "pointerdown,touchstart,mousedown", Zo), On(Pt, lt, "pointerup,touchend,mouseup", Qo), Qn.kill(), En(Pt);
            for (var n = 0; n < q.length; n += 3) Cn(Pt, q[n], q[n + 1]), Cn(Pt, q[n], q[n + 2])
        }, o.enable = function() {
            if (G = window, lt = document, He = lt.documentElement, nt = lt.body, D && (pn = D.utils.toArray, tn = D.utils.clamp, Ks = D.core.context || je, bs = D.core.suppressOverwrites || je, wo = G.history.scrollRestoration || "auto", Js = G.pageYOffset, D.core.globals("ScrollTrigger", o), nt)) {
                Xr = 1, xr = document.createElement("div"), xr.style.height = "100vh", xr.style.position = "absolute", Xl(), Gu(), vt.register(D), o.isTouch = vt.isTouch, yi = vt.isTouch && /(iPad|iPhone|iPod|Mac)/g.test(navigator.userAgent), Gs = vt.isTouch === 1, At(G, "wheel", ur), Pl = [G, lt, He, nt], D.matchMedia ? (o.matchMedia = function(l) {
                    var c = D.matchMedia(),
                        u;
                    for (u in l) c.add(u, l[u]);
                    return c
                }, D.addEventListener("matchMediaInit", function() {
                    return Eo()
                }), D.addEventListener("matchMediaRevert", function() {
                    return Hl()
                }), D.addEventListener("matchMedia", function() {
                    Vi(0, 1), ir("matchMedia")
                }), D.matchMedia("(orientation: portrait)", function() {
                    return ks(), ks
                })) : console.warn("Requires GSAP 3.11.0 or later"), ks(), At(lt, "scroll", ur);
                var e = nt.style,
                    i = e.borderTopStyle,
                    n = D.core.Animation.prototype,
                    s, a;
                for (n.revert || Object.defineProperty(n, "revert", {
                        value: function() {
                            return this.time(-.01, !0)
                        }
                    }), e.borderTopStyle = "solid", s = ai(nt), Et.m = Math.round(s.top + Et.sc()) || 0, ie.m = Math.round(s.left + ie.sc()) || 0, i ? e.borderTopStyle = i : e.removeProperty("border-top-style"), Sn = setInterval(ea, 250), D.delayedCall(.5, function() {
                        return kn = 0
                    }), At(lt, "touchcancel", je), At(nt, "touchstart", je), On(At, lt, "pointerdown,touchstart,mousedown", Zo), On(At, lt, "pointerup,touchend,mouseup", Qo), js = D.utils.checkPrefix("transform"), Wn.push(js), fr = Ht(), Qn = D.delayedCall(.2, Vi).pause(), hr = [lt, "visibilitychange", function() {
                        var l = G.innerWidth,
                            c = G.innerHeight;
                        lt.hidden ? ($o = l, jo = c) : ($o !== l || jo !== c) && $r()
                    }, lt, "DOMContentLoaded", Vi, G, "load", Vi, G, "resize", $r], En(At), H.forEach(function(l) {
                        return l.enable(0, 1)
                    }), a = 0; a < q.length; a += 3) Cn(Pt, q[a], q[a + 1]), Cn(Pt, q[a], q[a + 2])
            }
        }, o.config = function(e) {
            "limitCallbacks" in e && (xs = !!e.limitCallbacks);
            var i = e.syncInterval;
            i && clearInterval(Sn) || (Sn = i) && setInterval(ea, i), "ignoreMobileResize" in e && (Gs = o.isTouch === 1 && e.ignoreMobileResize), "autoRefreshEvents" in e && (En(Pt) || En(At, e.autoRefreshEvents || "none"), Ml = (e.autoRefreshEvents + "").indexOf("resize") === -1)
        }, o.scrollerProxy = function(e, i) {
            var n = se(e),
                s = q.indexOf(n),
                a = tr(n);
            ~s && q.splice(s, a ? 6 : 2), i && (a ? Je.unshift(G, i, nt, i, He, i) : Je.unshift(n, i))
        }, o.clearMatchMedia = function(e) {
            H.forEach(function(i) {
                return i._ctx && i._ctx.query === e && i._ctx.kill(!0, !0)
            })
        }, o.isInViewport = function(e, i, n) {
            var s = (ge(e) ? se(e) : e).getBoundingClientRect(),
                a = s[n ? Gi : Ki] * i || 0;
            return n ? s.right - a > 0 && s.left + a < G.innerWidth : s.bottom - a > 0 && s.top + a < G.innerHeight
        }, o.positionInViewport = function(e, i, n) {
            ge(e) && (e = se(e));
            var s = e.getBoundingClientRect(),
                a = s[n ? Gi : Ki],
                l = i == null ? a / 2 : i in es ? es[i] * a : ~i.indexOf("%") ? parseFloat(i) * a / 100 : parseFloat(i) || 0;
            return n ? (s.left + l) / G.innerWidth : (s.top + l) / G.innerHeight
        }, o.killAll = function(e) {
            if (H.slice(0).forEach(function(n) {
                    return n.vars.id !== "ScrollSmoother" && n.kill()
                }), e !== !0) {
                var i = er.killAll || [];
                er = {}, i.forEach(function(n) {
                    return n()
                })
            }
        }, o
    }();
V.version = "3.12.5";
V.saveStyles = function(o) {
    return o ? pn(o).forEach(function(t) {
        if (t && t.style) {
            var r = _e.indexOf(t);
            r >= 0 && _e.splice(r, 5), _e.push(t, t.style.cssText, t.getBBox && t.getAttribute("transform"), D.core.getCache(t), Ks())
        }
    }) : _e
};
V.revert = function(o, t) {
    return Eo(!o, t)
};
V.create = function(o, t) {
    return new V(o, t)
};
V.refresh = function(o) {
    return o ? $r() : (fr || V.register()) && Vi(!0)
};
V.update = function(o) {
    return ++q.cache && ui(o === !0 ? 2 : 0)
};
V.clearScrollMemory = Vl;
V.maxScroll = function(o, t) {
    return Ze(o, t ? ie : Et)
};
V.getScrollFunc = function(o, t) {
    return Ai(se(o), t ? ie : Et)
};
V.getById = function(o) {
    return Qs[o]
};
V.getAll = function() {
    return H.filter(function(o) {
        return o.vars.id !== "ScrollSmoother"
    })
};
V.isScrolling = function() {
    return !!De
};
V.snapDirectional = ko;
V.addEventListener = function(o, t) {
    var r = er[o] || (er[o] = []);
    ~r.indexOf(t) || r.push(t)
};
V.removeEventListener = function(o, t) {
    var r = er[o],
        e = r && r.indexOf(t);
    e >= 0 && r.splice(e, 1)
};
V.batch = function(o, t) {
    var r = [],
        e = {},
        i = t.interval || .016,
        n = t.batchMax || 1e9,
        s = function(c, u) {
            var h = [],
                d = [],
                f = D.delayedCall(i, function() {
                    u(h, d), h = [], d = []
                }).pause();
            return function(_) {
                h.length || f.restart(!0), h.push(_.trigger), d.push(_), n <= h.length && f.progress(1)
            }
        },
        a;
    for (a in t) e[a] = a.substr(0, 2) === "on" && re(t[a]) && a !== "onRefreshInit" ? s(a, t[a]) : t[a];
    return re(n) && (n = n(), At(V, "refresh", function() {
        return n = t.batchMax()
    })), pn(o).forEach(function(l) {
        var c = {};
        for (a in e) c[a] = e[a];
        c.trigger = l, r.push(V.create(c))
    }), r
};
var aa = function(t, r, e, i) {
        return r > i ? t(i) : r < 0 && t(0), e > i ? (i - r) / (e - r) : e < 0 ? r / (r - e) : 1
    },
    Os = function o(t, r) {
        r === !0 ? t.style.removeProperty("touch-action") : t.style.touchAction = r === !0 ? "auto" : r ? "pan-" + r + (vt.isTouch ? " pinch-zoom" : "") : "none", t === He && o(nt, r)
    },
    Rn = {
        auto: 1,
        scroll: 1
    },
    lf = function(t) {
        var r = t.event,
            e = t.target,
            i = t.axis,
            n = (r.changedTouches ? r.changedTouches[0] : r).target,
            s = n._gsap || D.core.getCache(n),
            a = Ht(),
            l;
        if (!s._isScrollT || a - s._isScrollT > 2e3) {
            for (; n && n !== nt && (n.scrollHeight <= n.clientHeight && n.scrollWidth <= n.clientWidth || !(Rn[(l = Pe(n)).overflowY] || Rn[l.overflowX]));) n = n.parentNode;
            s._isScroll = n && n !== e && !tr(n) && (Rn[(l = Pe(n)).overflowY] || Rn[l.overflowX]), s._isScrollT = a
        }(s._isScroll || i === "x") && (r.stopPropagation(), r._gsapAllow = !0)
    },
    ql = function(t, r, e, i) {
        return vt.create({
            target: t,
            capture: !0,
            debounce: !1,
            lockAxis: !0,
            type: r,
            onWheel: i = i && lf,
            onPress: i,
            onDrag: i,
            onScroll: i,
            onEnable: function() {
                return e && At(lt, vt.eventTypes[0], ca, !1, !0)
            },
            onDisable: function() {
                return Pt(lt, vt.eventTypes[0], ca, !0)
            }
        })
    },
    cf = /(input|label|select|textarea)/i,
    la, ca = function(t) {
        var r = cf.test(t.target.tagName);
        (r || la) && (t._gsapAllow = !0, la = r)
    },
    uf = function(t) {
        Wi(t) || (t = {}), t.preventDefault = t.isNormalizer = t.allowClicks = !0, t.type || (t.type = "wheel,touch"), t.debounce = !!t.debounce, t.id = t.id || "normalizer";
        var r = t,
            e = r.normalizeScrollX,
            i = r.momentum,
            n = r.allowNestedScroll,
            s = r.onRelease,
            a, l, c = se(t.target) || He,
            u = D.core.globals().ScrollSmoother,
            h = u && u.get(),
            d = yi && (t.content && se(t.content) || h && t.content !== !1 && !h.smooth() && h.content()),
            f = Ai(c, Et),
            _ = Ai(c, ie),
            p = 1,
            v = (vt.isTouch && G.visualViewport ? G.visualViewport.scale * G.visualViewport.width : G.outerWidth) / G.innerWidth,
            T = 0,
            b = re(i) ? function() {
                return i(a)
            } : function() {
                return i || 2.8
            },
            E, g, y = ql(c, t.type, !0, n),
            S = function() {
                return g = !1
            },
            w = je,
            C = je,
            k = function() {
                l = Ze(c, Et), C = tn(yi ? 1 : 0, l), e && (w = tn(0, Ze(c, ie))), E = Zi
            },
            O = function() {
                d._gsap.y = Ur(parseFloat(d._gsap.y) + f.offset) + "px", d.style.transform = "matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, " + parseFloat(d._gsap.y) + ", 0, 1)", f.offset = f.cacheID = 0
            },
            L = function() {
                if (g) {
                    requestAnimationFrame(S);
                    var $ = Ur(a.deltaY / 2),
                        J = C(f.v - $);
                    if (d && J !== f.v + f.offset) {
                        f.offset = J - f.v;
                        var m = Ur((parseFloat(d && d._gsap.y) || 0) - f.offset);
                        d.style.transform = "matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, " + m + ", 0, 1)", d._gsap.y = m + "px", f.cacheID = q.cache, ui()
                    }
                    return !0
                }
                f.offset && O(), g = !0
            },
            P, W, N, X, B = function() {
                k(), P.isActive() && P.vars.scrollY > l && (f() > l ? P.progress(1) && f(l) : P.resetTo("scrollY", l))
            };
        return d && D.set(d, {
            y: "+=0"
        }), t.ignoreCheck = function(M) {
            return yi && M.type === "touchmove" && L() || p > 1.05 && M.type !== "touchstart" || a.isGesturing || M.touches && M.touches.length > 1
        }, t.onPress = function() {
            g = !1;
            var M = p;
            p = Ur((G.visualViewport && G.visualViewport.scale || 1) / v), P.pause(), M !== p && Os(c, p > 1.01 ? !0 : e ? !1 : "x"), W = _(), N = f(), k(), E = Zi
        }, t.onRelease = t.onGestureStart = function(M, $) {
            if (f.offset && O(), !$) X.restart(!0);
            else {
                q.cache++;
                var J = b(),
                    m, tt;
                e && (m = _(), tt = m + J * .05 * -M.velocityX / .227, J *= aa(_, m, tt, Ze(c, ie)), P.vars.scrollX = w(tt)), m = f(), tt = m + J * .05 * -M.velocityY / .227, J *= aa(f, m, tt, Ze(c, Et)), P.vars.scrollY = C(tt), P.invalidate().duration(J).play(.01), (yi && P.vars.scrollY >= l || m >= l - 1) && D.to({}, {
                    onUpdate: B,
                    duration: J
                })
            }
            s && s(M)
        }, t.onWheel = function() {
            P._ts && P.pause(), Ht() - T > 1e3 && (E = 0, T = Ht())
        }, t.onChange = function(M, $, J, m, tt) {
            if (Zi !== E && k(), $ && e && _(w(m[2] === $ ? W + (M.startX - M.x) : _() + $ - m[1])), J) {
                f.offset && O();
                var qt = tt[2] === J,
                    Le = qt ? N + M.startY - M.y : f() + J - tt[1],
                    ft = C(Le);
                qt && Le !== ft && (N += ft - Le), f(ft)
            }(J || $) && ui()
        }, t.onEnable = function() {
            Os(c, e ? !1 : "x"), V.addEventListener("refresh", B), At(G, "resize", B), f.smooth && (f.target.style.scrollBehavior = "auto", f.smooth = _.smooth = !1), y.enable()
        }, t.onDisable = function() {
            Os(c, !0), Pt(G, "resize", B), V.removeEventListener("refresh", B), y.kill()
        }, t.lockAxis = t.lockAxis !== !1, a = new vt(t), a.iOS = yi, yi && !f() && f(1), yi && D.ticker.add(je), X = a._dc, P = D.to(a, {
            ease: "power4",
            paused: !0,
            inherit: !1,
            scrollX: e ? "+=0.1" : "+=0",
            scrollY: "+=0.1",
            modifiers: {
                scrollY: Ul(f, f(), function() {
                    return P.pause()
                })
            },
            onUpdate: ui,
            onComplete: X.vars.onComplete
        }), a
    };
V.sort = function(o) {
    return H.sort(o || function(t, r) {
        return (t.vars.refreshPriority || 0) * -1e6 + t.start - (r.start + (r.vars.refreshPriority || 0) * -1e6)
    })
};
V.observe = function(o) {
    return new vt(o)
};
V.normalizeScroll = function(o) {
    if (typeof o > "u") return Qt;
    if (o === !0 && Qt) return Qt.enable();
    if (o === !1) {
        Qt && Qt.kill(), Qt = o;
        return
    }
    var t = o instanceof vt ? o : uf(o);
    return Qt && Qt.target === t.target && Qt.kill(), tr(t.target) && (Qt = t), t
};
V.core = {
    _getVelocityProp: $s,
    _inputObserver: ql,
    _scrollers: q,
    _proxies: Je,
    bridge: {
        ss: function() {
            De || ir("scrollStart"), De = Ht()
        },
        ref: function() {
            return Wt
        }
    }
};
Il() && D.registerPlugin(V);
const ff = o => {
        const t = o.target.innerText,
            r = document.createElement("textarea");
        r.width = "1px", r.height = "1px", r.background = "transparents", r.value = t, document.body.append(r), r.select(), document.execCommand("copy"), document.body.removeChild(r)
    },
    hf = (o, t) => {
        let r = -1;
        const e = o == null ? 0 : o.length,
            i = new Array(e);
        for (; ++r < e;) i[r] = t(o[r], r, o);
        return i
    };

function eo(o, t) {
    return o instanceof window.HTMLElement ? [t(o)] : hf(o, t)
}
const df = o => {
    const t = new Set;
    do
        for (const r of Reflect.ownKeys(o)) t.add([o, r]); while ((o = Reflect.getPrototypeOf(o)) && o !== Object.prototype);
    return t
};

function pf(o, {
    include: t,
    exclude: r
} = {}) {
    const e = i => {
        const n = s => typeof s == "string" ? i === s : s.test(i);
        return t ? t.some(n) : r ? !r.some(n) : !0
    };
    for (const [i, n] of df(o.constructor.prototype)) {
        if (n === "constructor" || !e(n)) continue;
        const s = Reflect.getOwnPropertyDescriptor(i, n);
        s && typeof s.value == "function" && (o[n] = o[n].bind(o))
    }
    return o
}
class _f {
    constructor({
        element: t,
        elements: r
    }) {
        pf(this), this.selector = t, this.selectorChildren = {
            ...r
        }, this.create()
    }
    create() {
        this.selector instanceof HTMLElement ? this.element = this.selector : this.element = document.querySelector(this.selector), this.elements = {}, Object.keys(this.selectorChildren).forEach(t => {
            const r = this.selectorChildren[t];
            r instanceof HTMLElement || r instanceof NodeList || Array.isArray(r) ? this.elements[t] = r : (this.elements[t] = this.element.querySelectorAll(r), this.elements[t].length === 0 ? this.elements[t] = null : this.elements[t].length === 1 && (this.elements[t] = this.element.querySelector(r)))
        })
    }
}
class gf extends _f {
    constructor() {
        super({
            element: "body",
            elements: {
                hour: "[data-hour]",
                minute: "[data-minute]"
            }
        }), this.updateTime(), this.oldHour = this.formattedTime.hourValue, this.oldMinute = this.formattedTime.minuteValue, setInterval(this.updateTime, 1e3)
    }
    get currentTime() {
        const t = {
            hour: "numeric",
            minute: "numeric",
            timeZone: "America/Edmonton"
        };
        return new Intl.DateTimeFormat([], t).format(new Date)
    }
    get formattedTime() {
        const r = this.currentTime.split(":"),
            e = r[0],
            i = r[1];
        return {
            hourValue: e,
            minuteValue: i
        }
    }
    updateTime() {
        const {
            hour: t,
            minute: r
        } = this.elements, {
            hourValue: e,
            minuteValue: i
        } = this.formattedTime;
        eo(t, n => {
            this.oldHour !== e && (n.classList.add("flash"), setTimeout(() => {
                n.innerHTML = e
            }, 500), setTimeout(() => {
                n.classList.remove("flash")
            }, 1e3))
        }), eo(r, n => {
            this.oldMinute !== i && (n.classList.add("flash"), setTimeout(() => {
                n.innerHTML = String(i).slice(0, 2)
            }, 500), setTimeout(() => {
                n.classList.remove("flash")
            }, 1e3))
        }), this.oldHour = e, this.oldMinute = i
    }
}
const mf = document.querySelectorAll(".contact-scroll"),
    vf = document.getElementById("js-footer"),
    $l = document.querySelector("[data-scroll-container]"),
    yf = document.querySelector("button.email"),
    ua = document.querySelector(".to-copy span");
new gf;
pe.registerPlugin(V);
const Sr = new Mc({
    el: $l,
    smooth: !0,
    lerp: .06,
    tablet: {
        breakpoint: 768
    }
});
setTimeout(() => {
    Sr.update()
}, 1e3);
Sr.on("scroll", V.update);
V.scrollerProxy(Sr.el, {
    scrollTop(o) {
        return arguments.length ? Sr.scrollTo(o, 0, 0) : Sr.scroll.instance.scroll.y
    },
    getBoundingClientRect() {
        return {
            top: 0,
            left: 0,
            width: window.innerWidth,
            height: window.innerHeight
        }
    }
});
class wf {
    constructor(t) {
        this.locomotive = t, this.heroTextAnimation(), this.homeIntro(), this.homeAnimations(), this.homeActions()
    }
    homeActions() {
        eo(mf, t => {
            t.onclick = () => {
                this.locomotive.scrollTo(vf)
            }
        }), yf.addEventListener("click", t => {
            ff(t), ua.textContent = "copied", setTimeout(() => {
                ua.textContent = "Click To Copy"
            }, 2e3)
        })
    }
    homeIntro() {
        const t = pe.timeline();
        pe.to($l, {
            autoAlpha: 1
        }), t.from(".home__nav", {
            duration: .5,
            delay: .3,
            opacity: 0,
            yPercent: -100,
            ease: "power4.out"
        }).from(".hero__title [title-overflow]", {
            duration: .7,
            yPercent: 100,
            stagger: {
                amount: .2
            },
            ease: "power4.out"
        }).from(".hero__title .bottom__right", {
            duration: 1,
            yPercent: 100,
            opacity: 0,
            ease: "power4.out"
        }, "<20%").set(".hero__title .overflow", {
            overflow: "unset"
        }).from(".hero__title .mobile", {
            duration: .7,
            yPercent: 100,
            stagger: {
                amount: .2
            },
            ease: "power4.out"
        }, "-=1.4")
    }
    homeAnimations() {
        pe.to(".home__projects__line", {
            autoAlpha: 1
        }), pe.utils.toArray(".home__projects__line").forEach(t => {
            const r = t.querySelector("span");
            pe.from(r, {
                duration: 1.5,
                scrollTrigger: {
                    trigger: t,
                    scroller: "[data-scroll-container]"
                },
                scaleX: 0
            })
        }), pe.utils.toArray("[data-fade-in]").forEach(t => {
            pe.from(t, {
                scrollTrigger: {
                    trigger: t,
                    scroller: "[data-scroll-container]"
                },
                duration: 1.5,
                yPercent: 100,
                opacity: 0,
                ease: "power4.out"
            })
        }), window.innerWidth <= 768 && (pe.utils.toArray(".home__projects__project").forEach(r => {
            const e = r.querySelector(".title__main"),
                i = r.querySelector(".project__link");
            pe.from([e, i], {
                scrollTrigger: {
                    trigger: r,
                    scroller: "[data-scroll-container]"
                },
                duration: 1.5,
                yPercent: 100,
                stagger: {
                    amount: .2
                },
                ease: "power4.out"
            })
        }), pe.timeline({
            defaults: {
                ease: "power1.out"
            },
            scrollTrigger: {
                trigger: ".home__awards",
                scroller: "[data-scroll-container]"
            }
        }).from(".awards__title span", {
            duration: 1,
            opacity: 0,
            yPercent: 100,
            stagger: {
                amount: .2
            }
        }))
    }
    heroTextAnimation() {
        pe.to(".hero__title__dash.desktop", {
            scrollTrigger: {
                trigger: ".hero__title",
                scroller: "[data-scroll-container]",
                scrub: !0,
                start: "-8% 9%",
                end: "110% 20%"
            },
            scaleX: 4,
            ease: "none"
        })
    }
}
new wf(Sr);

function sendDataToFlask() {
    // Get values from input fields
    const inputA = document.getElementById('inputA').value;
    const inputB = document.getElementById('inputB').value;
    const inputC = document.getElementById('inputC').value;
    const inputD = document.getElementById('inputD').value;

    // Store values in an array
    const dataArray = [inputB, inputA, inputD, inputC];

    console.log(dataArray)

    // Send the array to Flask using fetch
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: dataArray }), // Send the array as JSON
    })
    .then(response => response.json()) // Parse the JSON response
    .then(data => {

        console.log("Received data from Flask:", data);
        // Assuming data is an array of coordinate objects
        // e.g. [{ coords: [lat, lon], name: "Point 1" }, ... ]

        // Store the received coordinates in localStorage
        localStorage.setItem('locations', JSON.stringify(data));

        // Redirect to map.html
        window.location.href = "map.html";

    })
    .catch((error) => {
        console.error('Error:', error); // Handle errors
    });
}

window.sendDataToFlask = sendDataToFlask;